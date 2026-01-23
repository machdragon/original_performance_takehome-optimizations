"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(
        self,
        enable_debug: bool = False,
        assume_zero_indices: bool = True,
        max_special_level: int = -1,
        max_arith_level: int = -1,
        enable_prefetch: bool = False,
    ):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.enable_debug = enable_debug
        self.assume_zero_indices = assume_zero_indices
        self.max_special_level = max_special_level
        self.max_arith_level = max_arith_level
        self.enable_prefetch = enable_prefetch

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _slot_reads_writes(self, engine, slot):
        reads = set()
        writes = set()
        barrier = False

        def add_vec(base):
            return set(range(base, base + VLEN))

        if engine == "debug":
            return reads, writes, True

        if engine == "alu":
            _, dest, a1, a2 = slot
            reads.update([a1, a2])
            writes.add(dest)
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                _, dest, src = slot
                reads.add(src)
                writes.update(add_vec(dest))
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                reads.update(add_vec(a))
                reads.update(add_vec(b))
                reads.update(add_vec(c))
                writes.update(add_vec(dest))
            else:
                _, dest, a1, a2 = slot
                reads.update(add_vec(a1))
                reads.update(add_vec(a2))
                writes.update(add_vec(dest))
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    reads.add(addr)
                    writes.add(dest)
                case ("load_offset", dest, addr, offset):
                    reads.add(addr + offset)
                    writes.add(dest + offset)
                case ("vload", dest, addr):
                    reads.add(addr)
                    writes.update(add_vec(dest))
                case ("const", dest, _val):
                    writes.add(dest)
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    reads.update([addr, src])
                case ("vstore", addr, src):
                    reads.add(addr)
                    reads.update(add_vec(src))
        elif engine == "flow":
            op = slot[0]
            if op in ("halt", "pause", "jump", "jump_indirect", "cond_jump", "cond_jump_rel"):
                barrier = True
            match slot:
                case ("select", dest, cond, a, b):
                    reads.update([cond, a, b])
                    writes.add(dest)
                case ("add_imm", dest, a, _imm):
                    reads.add(a)
                    writes.add(dest)
                case ("vselect", dest, cond, a, b):
                    reads.update(add_vec(cond))
                    reads.update(add_vec(a))
                    reads.update(add_vec(b))
                    writes.update(add_vec(dest))
                case ("trace_write", val):
                    reads.add(val)
                case ("coreid", dest):
                    writes.add(dest)

        return reads, writes, barrier

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        instrs = []
        bundle = {}
        bundle_reads = set()
        bundle_writes = set()

        slots_info = []
        for engine, slot in slots:
            reads, writes, barrier = self._slot_reads_writes(engine, slot)
            slots_info.append(
                {
                    "engine": engine,
                    "slot": slot,
                    "reads": reads,
                    "writes": writes,
                    "barrier": barrier,
                }
            )

        def flush():
            nonlocal bundle, bundle_reads, bundle_writes
            if bundle:
                instrs.append(bundle)
            bundle = {}
            bundle_reads = set()
            bundle_writes = set()

        def bundle_loads():
            return len(bundle.get("load", []))

        def engine_full(engine_name):
            return len(bundle.get(engine_name, [])) >= SLOT_LIMITS[engine_name]

        def add_to_bundle(info):
            bundle.setdefault(info["engine"], []).append(info["slot"])
            bundle_reads.update(info["reads"])
            bundle_writes.update(info["writes"])

        def can_add_now(info):
            if engine_full(info["engine"]):
                return False
            if info["reads"] & bundle_writes:
                return False
            if info["writes"] & bundle_writes:
                return False
            return True

        def find_pullable_slot(
            start_idx,
            future_writes,
            engine_filter=None,
            lookahead=16,
            init_skipped_reads=None,
            init_skipped_writes=None,
        ):
            skipped_writes = (
                set() if init_skipped_writes is None else set(init_skipped_writes)
            )
            skipped_reads = (
                set() if init_skipped_reads is None else set(init_skipped_reads)
            )
            for j in range(start_idx, min(len(slots_info), start_idx + lookahead)):
                info = slots_info[j]
                if info is None:
                    continue
                if info["barrier"]:
                    break
                reads = info["reads"]
                writes = info["writes"]
                if engine_filter is not None and info["engine"] != engine_filter:
                    skipped_reads.update(reads)
                    skipped_writes.update(writes)
                    continue
                if engine_full(info["engine"]):
                    skipped_reads.update(reads)
                    skipped_writes.update(writes)
                    continue
                if reads & skipped_writes:
                    pass
                elif writes & skipped_writes:
                    pass
                elif writes & skipped_reads:
                    pass
                elif reads & future_writes:
                    pass
                elif writes & future_writes:
                    pass
                else:
                    return j
                skipped_reads.update(reads)
                skipped_writes.update(writes)
            return None

        i = 0
        while i < len(slots_info):
            info = slots_info[i]
            if info is None:
                i += 1
                continue
            engine = info["engine"]
            slot = info["slot"]
            reads = info["reads"]
            writes = info["writes"]
            barrier = info["barrier"]

            if barrier:
                flush()
                instrs.append({engine: [slot]})
                i += 1
                continue

            if engine != "load" and bundle_loads() < SLOT_LIMITS["load"]:
                while bundle_loads() < SLOT_LIMITS["load"]:
                    future_writes = bundle_writes | writes
                    pull_idx = find_pullable_slot(
                        i + 1,
                        future_writes,
                        engine_filter="load",
                    )
                    if pull_idx is None:
                        break
                    pull = slots_info[pull_idx]
                    slots_info[pull_idx] = None
                    add_to_bundle(pull)

            if not can_add_now(info):
                if bundle:
                    blocked_reads = reads
                    blocked_writes = writes
                    while True:
                        pull_idx = find_pullable_slot(
                            i + 1,
                            bundle_writes,
                            init_skipped_reads=blocked_reads,
                            init_skipped_writes=blocked_writes,
                        )
                        if pull_idx is None:
                            break
                        pull = slots_info[pull_idx]
                        slots_info[pull_idx] = None
                        add_to_bundle(pull)
                    flush()
                else:
                    flush()
                continue

            add_to_bundle(info)
            i += 1

        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, length=VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                slots.append(
                    ("alu", ("*", tmp1, val_hash_addr, self.scratch_const(factor)))
                )
                slots.append(
                    ("alu", ("+", val_hash_addr, tmp1, self.scratch_const(val1)))
                )
            else:
                slots.append(
                    ("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1)))
                )
                slots.append(
                    ("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3)))
                )
                slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            if self.enable_debug:
                slots.append(
                    ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
                )

        return slots

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2):
        slots = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                slots.append(
                    (
                        "valu",
                        (
                            "multiply_add",
                            val_hash_addr,
                            val_hash_addr,
                            self.scratch_vconst(factor),
                            self.scratch_vconst(val1),
                        ),
                    )
                )
            else:
                slots.append(
                    ("valu", (op1, tmp1, val_hash_addr, self.scratch_vconst(val1)))
                )
                slots.append(
                    ("valu", (op3, tmp2, val_hash_addr, self.scratch_vconst(val3)))
                )
                slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        write_indices: bool = True,
        write_indicies: bool | None = None,
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation that uses SIMD for the batch dimension.
        """
        if write_indicies is not None:
            write_indices = write_indicies
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Cache indices and values in scratch across rounds to avoid repeated loads/stores.
        idx_cache = self.alloc_scratch("idx_cache", batch_size)
        val_cache = self.alloc_scratch("val_cache", batch_size)
        val_load_ptr = self.alloc_scratch("val_load_ptr")
        self.add("alu", ("+", val_load_ptr, self.scratch["inp_values_p"], zero_const))
        # Scratch starts zeroed; when indices are assumed zero we can skip loads.
        idx_load_ptr = None
        if not self.assume_zero_indices:
            idx_load_ptr = self.alloc_scratch("idx_load_ptr")
            self.add(
                "alu", ("+", idx_load_ptr, self.scratch["inp_indices_p"], zero_const)
            )
        vec_end = batch_size - (batch_size % VLEN)
        for i in range(0, vec_end, VLEN):
            if idx_load_ptr is not None:
                self.add("load", ("vload", idx_cache + i, idx_load_ptr))
            self.add("load", ("vload", val_cache + i, val_load_ptr))
            if idx_load_ptr is not None:
                self.add("flow", ("add_imm", idx_load_ptr, idx_load_ptr, VLEN))
            self.add("flow", ("add_imm", val_load_ptr, val_load_ptr, VLEN))
        for i in range(vec_end, batch_size):
            if idx_load_ptr is not None:
                self.add("load", ("load", idx_cache + i, idx_load_ptr))
            self.add("load", ("load", val_cache + i, val_load_ptr))
            if idx_load_ptr is not None:
                self.add("flow", ("add_imm", idx_load_ptr, idx_load_ptr, 1))
            self.add("flow", ("add_imm", val_load_ptr, val_load_ptr, 1))

        enable_arith = self.max_arith_level >= 2

        # Vector scratch registers and broadcasted constants.
        v_node_val = [
            self.alloc_scratch("v_node_val_0", VLEN),
            self.alloc_scratch("v_node_val_1", VLEN),
        ]
        v_addr = [
            self.alloc_scratch("v_addr_0", VLEN),
            self.alloc_scratch("v_addr_1", VLEN),
        ]
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_tmp4 = None
        if enable_arith:
            v_tmp4 = self.alloc_scratch("v_tmp4", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_three = None
        if enable_arith:
            v_three = self.alloc_scratch("v_three", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_values_p", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        if enable_arith:
            self.add("valu", ("vbroadcast", v_three, self.scratch_const(3)))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))
        for _, val1, _, _, val3 in HASH_STAGES:
            self.scratch_vconst(val1)
            self.scratch_vconst(val3)

        # Block size 8 fits 32 vectors (no remainder) and reduces block overhead.
        block_size = 8
        # Double buffers for pipelining
        v_node_block = [
            self.alloc_scratch("v_node_block_A", block_size * VLEN),
            self.alloc_scratch("v_node_block_B", block_size * VLEN),
        ]
        v_tmp1_block = self.alloc_scratch("v_tmp1_block", block_size * VLEN)
        v_tmp2_block = self.alloc_scratch("v_tmp2_block", block_size * VLEN)
        v_tmp3_block = self.alloc_scratch("v_tmp3_block", block_size * VLEN)
        v_tmp4_block = None
        if enable_arith:
            v_tmp4_block = self.alloc_scratch("v_tmp4_block", block_size * VLEN)

        max_special = min(self.max_special_level, forest_height)
        use_special = self.assume_zero_indices and max_special >= 0
        level_vec_base = []
        level_sizes = []
        level_starts = []
        level_tmp = None
        if use_special:
            level_total = (1 << (max_special + 1)) - 1
            level_values = self.alloc_scratch("level_values", level_total)
            level_vecs = self.alloc_scratch("level_vecs", level_total * VLEN)
            level_tmp = self.alloc_scratch("level_tmp_vecs", (1 << max_special) * VLEN)
            level_addr = self.alloc_scratch("level_addr")
            offset = 0
            for level in range(max_special + 1):
                level_size = 1 << level
                level_start = level_size - 1
                level_vec_base.append(level_vecs + offset * VLEN)
                level_sizes.append(level_size)
                level_starts.append(level_start)
                # Load level values once and broadcast into vectors.
                self.add(
                    "alu",
                    (
                        "+",
                        level_addr,
                        self.scratch["forest_values_p"],
                        self.scratch_const(level_start),
                    ),
                )
                vec_chunks = level_size // VLEN
                tail = level_size % VLEN
                for _ in range(vec_chunks):
                    self.add("load", ("vload", level_values + offset, level_addr))
                    self.add("flow", ("add_imm", level_addr, level_addr, VLEN))
                    offset += VLEN
                for _ in range(tail):
                    self.add("load", ("load", level_values + offset, level_addr))
                    self.add("flow", ("add_imm", level_addr, level_addr, 1))
                    offset += 1
                for p in range(level_size):
                    self.add(
                        "valu",
                        (
                            "vbroadcast",
                            level_vecs + (offset - level_size + p) * VLEN,
                            level_values + (offset - level_size + p),
                        ),
                    )

        def interleave_slots(hash_slots, load_slots):
            """Interleave hash and load slots, emitting loads in pairs."""
            if not load_slots:
                return hash_slots
            if not hash_slots:
                return load_slots
            combined = []
            h_len = len(hash_slots)
            l_len = len(load_slots)
            # Emit 2 loads per spacing to fill both load slots
            spacing = max(1, h_len // ((l_len + 1) // 2 + 1))
            h_i = 0
            l_i = 0
            since_load = 0
            while h_i < h_len or l_i < l_len:
                if h_i < h_len:
                    combined.append(hash_slots[h_i])
                    h_i += 1
                    since_load += 1
                if l_i < l_len and since_load >= spacing:
                    combined.append(load_slots[l_i])
                    l_i += 1
                    if l_i < l_len:
                        combined.append(load_slots[l_i])
                        l_i += 1
                    since_load = 0
                elif h_i >= h_len and l_i < l_len:
                    combined.append(load_slots[l_i])
                    l_i += 1
            return combined

        def vec_load_slots(
            vec_i,
            buf_idx,
            node_const=None,
            node_pair=None,
            node_arith=None,
            node_prefetch=None,
        ):
            slots = []
            if (
                node_const is not None
                or node_pair is not None
                or node_arith is not None
                or node_prefetch is not None
            ):
                return slots
            v_idx = idx_cache + vec_i
            v_addr_buf = v_addr[buf_idx]
            v_node_buf = v_node_val[buf_idx]
            slots.append(("valu", ("+", v_addr_buf, v_idx, v_forest_p)))
            for lane in range(VLEN):
                slots.append(("load", ("load_offset", v_node_buf, v_addr_buf, lane)))
            return slots

        def emit_level_select_slots(v_idx, buf_idx, node_arith):
            level = node_arith["level"]
            v_start = node_arith["v_start"]
            bases = node_arith["bases"]
            diffs = node_arith["diffs"]
            slots = []
            v_offset = v_addr[buf_idx]
            if level == 2:
                slots.append(("valu", ("-", v_offset, v_idx, v_start)))
                slots.append(("valu", ("&", v_tmp1, v_offset, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp2, v_tmp1, diffs[0], bases[0]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_tmp3, v_tmp1, diffs[1], bases[1]))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_one)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp3, v_tmp3, v_tmp2)))
                slots.append(("valu", ("multiply_add", v_tmp3, v_tmp1, v_tmp3, v_tmp2)))
                return slots, v_tmp3
            if level == 3:
                v_node_buf = v_node_val[buf_idx]
                slots.append(("valu", ("-", v_offset, v_idx, v_start)))
                slots.append(("valu", ("&", v_tmp1, v_offset, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp2, v_tmp1, diffs[0], bases[0]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_tmp3, v_tmp1, diffs[1], bases[1]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_tmp4, v_tmp1, diffs[2], bases[2]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_node_buf, v_tmp1, diffs[3], bases[3]))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_one)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp3, v_tmp3, v_tmp2)))
                slots.append(("valu", ("multiply_add", v_tmp2, v_tmp1, v_tmp3, v_tmp2)))
                slots.append(("valu", ("-", v_node_buf, v_node_buf, v_tmp4)))
                slots.append(
                    ("valu", ("multiply_add", v_node_buf, v_tmp1, v_node_buf, v_tmp4))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_two)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_node_buf, v_node_buf, v_tmp2)))
                slots.append(
                    ("valu", ("multiply_add", v_node_buf, v_tmp1, v_node_buf, v_tmp2))
                )
                return slots, v_node_buf
            if level == 4:
                v_node_buf = v_node_val[buf_idx]
                # Group-of-4 selection for q0..q3, then combine with bits 2/3.
                slots.append(("valu", ("-", v_offset, v_idx, v_start)))
                slots.append(("valu", ("&", v_tmp1, v_offset, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp2, v_tmp1, diffs[0], bases[0]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_tmp3, v_tmp1, diffs[1], bases[1]))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_one)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp3, v_tmp3, v_tmp2)))
                slots.append(("valu", ("multiply_add", v_tmp2, v_tmp1, v_tmp3, v_tmp2)))

                slots.append(("valu", ("&", v_tmp1, v_offset, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp3, v_tmp1, diffs[2], bases[2]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_tmp4, v_tmp1, diffs[3], bases[3]))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_one)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp4, v_tmp4, v_tmp3)))
                slots.append(("valu", ("multiply_add", v_tmp3, v_tmp1, v_tmp4, v_tmp3)))

                slots.append(("valu", (">>", v_tmp1, v_offset, v_two)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp3, v_tmp3, v_tmp2)))
                slots.append(("valu", ("multiply_add", v_tmp2, v_tmp1, v_tmp3, v_tmp2)))

                slots.append(("valu", ("&", v_tmp1, v_offset, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp3, v_tmp1, diffs[4], bases[4]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_tmp4, v_tmp1, diffs[5], bases[5]))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_one)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp4, v_tmp4, v_tmp3)))
                slots.append(("valu", ("multiply_add", v_tmp3, v_tmp1, v_tmp4, v_tmp3)))

                slots.append(("valu", ("&", v_tmp1, v_offset, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp4, v_tmp1, diffs[6], bases[6]))
                )
                slots.append(
                    ("valu", ("multiply_add", v_node_buf, v_tmp1, diffs[7], bases[7]))
                )
                slots.append(("valu", (">>", v_tmp1, v_offset, v_one)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_node_buf, v_node_buf, v_tmp4)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp4, v_tmp1, v_node_buf, v_tmp4))
                )

                slots.append(("valu", (">>", v_tmp1, v_offset, v_two)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp4, v_tmp4, v_tmp3)))
                slots.append(("valu", ("multiply_add", v_tmp3, v_tmp1, v_tmp4, v_tmp3)))

                slots.append(("valu", (">>", v_tmp1, v_offset, v_three)))
                slots.append(("valu", ("&", v_tmp1, v_tmp1, v_one)))
                slots.append(("valu", ("-", v_tmp3, v_tmp3, v_tmp2)))
                slots.append(("valu", ("multiply_add", v_tmp2, v_tmp1, v_tmp3, v_tmp2)))
                return slots, v_tmp2
            return slots, v_node_val[buf_idx]

        def vec_hash_slots(
            vec_i,
            buf_idx,
            round_idx,
            wrap_round,
            node_const=None,
            node_pair=None,
            node_arith=None,
            node_prefetch=None,
        ):
            slots = []
            v_idx = idx_cache + vec_i
            v_val = val_cache + vec_i
            if node_const is not None:
                v_node_buf = node_const
            elif node_pair is not None:
                node_right, node_diff = node_pair
                slots.append(("valu", ("&", v_tmp1, v_idx, v_one)))
                slots.append(
                    ("valu", ("multiply_add", v_tmp3, v_tmp1, node_diff, node_right))
                )
                v_node_buf = v_tmp3
            elif node_prefetch is not None:
                v_node_buf = node_prefetch + vec_i
            elif node_arith is not None:
                select_slots, v_node_buf = emit_level_select_slots(
                    v_idx, buf_idx, node_arith
                )
                slots.extend(select_slots)
            else:
                v_node_buf = v_node_val[buf_idx]
            if self.enable_debug:
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_idx,
                            [(round_idx, vec_i + lane, "idx") for lane in range(VLEN)],
                        ),
                    )
                )
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_val,
                            [(round_idx, vec_i + lane, "val") for lane in range(VLEN)],
                        ),
                    )
                )
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_node_buf,
                            [
                                (round_idx, vec_i + lane, "node_val")
                                for lane in range(VLEN)
                            ],
                        ),
                    )
                )
            slots.append(("valu", ("^", v_val, v_val, v_node_buf)))
            slots.extend(self.build_hash_vec(v_val, v_tmp1, v_tmp2))
            if self.enable_debug:
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_val,
                            [
                                (round_idx, vec_i + lane, "hashed_val")
                                for lane in range(VLEN)
                            ],
                        ),
                    )
                )
            if fast_wrap:
                if wrap_round:
                    slots.append(("valu", ("+", v_idx, v_zero, v_zero)))
                else:
                    slots.append(("valu", ("&", v_tmp3, v_val, v_one)))
                    slots.append(("valu", ("+", v_tmp3, v_tmp3, v_one)))
                    slots.append(("valu", ("multiply_add", v_idx, v_idx, v_two, v_tmp3)))
            else:
                slots.append(("valu", ("&", v_tmp3, v_val, v_one)))
                slots.append(("valu", ("+", v_tmp3, v_tmp3, v_one)))
                slots.append(("valu", ("+", v_idx, v_idx, v_idx)))
                slots.append(("valu", ("+", v_idx, v_idx, v_tmp3)))
            if self.enable_debug:
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_idx,
                            [
                                (round_idx, vec_i + lane, "next_idx")
                                for lane in range(VLEN)
                            ],
                        ),
                    )
                )
            if not fast_wrap:
                slots.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
                slots.append(("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero)))
            if self.enable_debug:
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_idx,
                            [
                                (round_idx, vec_i + lane, "wrapped_idx")
                                for lane in range(VLEN)
                            ],
                        ),
                    )
                )
            return slots

        def vec_special_slots(vec_i, buf_idx, level):
            slots = []
            v_idx = idx_cache + vec_i
            v_node_buf = v_node_val[buf_idx]
            v_offset = v_tmp3
            v_level_start = self.scratch_vconst(level_starts[level])
            slots.append(("valu", ("-", v_offset, v_idx, v_level_start)))
            current_base = level_vec_base[level]
            num = level_sizes[level]
            bit = 0
            temp_base = level_tmp
            while num > 1:
                v_bit = self.scratch_vconst(bit)
                slots.append(("valu", (">>", v_tmp1, v_offset, v_bit)))
                slots.append(("valu", ("&", v_tmp2, v_tmp1, v_one)))
                for pair in range(num // 2):
                    low = current_base + (2 * pair) * VLEN
                    high = low + VLEN
                    out = temp_base + pair * VLEN
                    slots.append(("flow", ("vselect", out, v_tmp2, high, low)))
                current_base = temp_base
                num //= 2
                bit += 1
            slots.append(("valu", ("+", v_node_buf, current_base, v_zero)))
            return slots

        def vec_block_load_slots(
            block_vecs,
            buf_idx,
            node_const=None,
            node_pair=None,
            node_arith=None,
            node_prefetch=None,
        ):
            """Load node values for a block into the specified buffer."""
            slots = []
            if (
                node_const is not None
                or node_pair is not None
                or node_arith is not None
                or node_prefetch is not None
            ):
                return slots
            node_buf = v_node_block[buf_idx]
            for bi, vec_i in enumerate(block_vecs):
                v_idx = idx_cache + vec_i
                slots.append(("valu", ("+", v_addr[0], v_idx, v_forest_p)))
                for lane in range(VLEN):
                    slots.append(
                        (
                            "load",
                            (
                                "load_offset",
                                node_buf + bi * VLEN,
                                v_addr[0],
                                lane,
                            ),
                        )
                    )
            return slots

        def vec_block_prefetch_slots(block_vecs, prefetch_base):
            """Load node values for a block into a persistent prefetch buffer."""
            slots = []
            for vec_i in block_vecs:
                v_idx = idx_cache + vec_i
                slots.append(("valu", ("+", v_addr[0], v_idx, v_forest_p)))
                for lane in range(VLEN):
                    slots.append(
                        ("load", ("load_offset", prefetch_base + vec_i, v_addr[0], lane))
                    )
            return slots

        def emit_level_select_block_slots(block_vecs, buf_idx, node_arith):
            level = node_arith["level"]
            v_start = node_arith["v_start"]
            bases = node_arith["bases"]
            diffs = node_arith["diffs"]
            slots = []
            node_buf = v_node_block[buf_idx]
            for bi, vec_i in enumerate(block_vecs):
                v_idx = idx_cache + vec_i
                # Offset is recomputed into bit each phase; could be cached to save VALU ops.
                bit = v_tmp1_block + bi * VLEN
                t0 = v_tmp2_block + bi * VLEN
                t1 = v_tmp3_block + bi * VLEN
                t2 = v_tmp4_block + bi * VLEN
                node = node_buf + bi * VLEN
                if level == 2:
                    slots.append(("valu", ("-", t0, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, t0, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t0, bit, diffs[0], bases[0]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", t1, bit, diffs[1], bases[1]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t1, t1, t0)))
                    slots.append(("valu", ("multiply_add", node, bit, t1, t0)))
                elif level == 3:
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t0, bit, diffs[0], bases[0]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", t1, bit, diffs[1], bases[1]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t1, t1, t0)))
                    slots.append(("valu", ("multiply_add", t0, bit, t1, t0)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t1, bit, diffs[2], bases[2]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", t2, bit, diffs[3], bases[3]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t2, t2, t1)))
                    slots.append(("valu", ("multiply_add", t1, bit, t2, t1)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_two)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t1, t1, t0)))
                    slots.append(("valu", ("multiply_add", node, bit, t1, t0)))
                else:
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t0, bit, diffs[0], bases[0]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", t1, bit, diffs[1], bases[1]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t1, t1, t0)))
                    slots.append(("valu", ("multiply_add", t0, bit, t1, t0)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t1, bit, diffs[2], bases[2]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", t2, bit, diffs[3], bases[3]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t2, t2, t1)))
                    slots.append(("valu", ("multiply_add", t1, bit, t2, t1)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_two)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t1, t1, t0)))
                    slots.append(("valu", ("multiply_add", t0, bit, t1, t0)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t1, bit, diffs[4], bases[4]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", t2, bit, diffs[5], bases[5]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t2, t2, t1)))
                    slots.append(("valu", ("multiply_add", t1, bit, t2, t1)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(
                        ("valu", ("multiply_add", t2, bit, diffs[6], bases[6]))
                    )
                    slots.append(
                        ("valu", ("multiply_add", node, bit, diffs[7], bases[7]))
                    )
                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_one)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", node, node, t2)))
                    slots.append(("valu", ("multiply_add", t2, bit, node, t2)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_two)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t2, t2, t1)))
                    slots.append(("valu", ("multiply_add", t1, bit, t2, t1)))

                    slots.append(("valu", ("-", bit, v_idx, v_start)))
                    slots.append(("valu", (">>", bit, bit, v_three)))
                    slots.append(("valu", ("&", bit, bit, v_one)))
                    slots.append(("valu", ("-", t1, t1, t0)))
                    slots.append(("valu", ("multiply_add", node, bit, t1, t0)))
            return slots, node_buf

        def vec_block_hash_only_slots(
            block_vecs,
            buf_idx,
            wrap_round,
            node_const=None,
            node_pair=None,
            node_arith=None,
            node_prefetch=None,
        ):
            """XOR, hash, and update indices using node values from specified buffer."""
            slots = []
            node_buf = v_node_block[buf_idx]
            # XOR with node values.
            if node_const is not None:
                for _bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(("valu", ("^", v_val, v_val, node_const)))
            elif node_pair is not None:
                node_right, node_diff = node_pair
                for bi, vec_i in enumerate(block_vecs):
                    v_idx = idx_cache + vec_i
                    slots.append(
                        ("valu", ("&", v_tmp1_block + bi * VLEN, v_idx, v_one))
                    )
                    slots.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                v_tmp3_block + bi * VLEN,
                                v_tmp1_block + bi * VLEN,
                                node_diff,
                                node_right,
                            ),
                        )
                    )
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(
                        ("valu", ("^", v_val, v_val, v_tmp3_block + bi * VLEN))
                    )
            elif node_prefetch is not None:
                for _bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(
                        ("valu", ("^", v_val, v_val, node_prefetch + vec_i))
                    )
            elif node_arith is not None:
                select_slots, node_base = emit_level_select_block_slots(
                    block_vecs, buf_idx, node_arith
                )
                slots.extend(select_slots)
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(
                        ("valu", ("^", v_val, v_val, node_base + bi * VLEN))
                    )
            else:
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(
                        ("valu", ("^", v_val, v_val, node_buf + bi * VLEN))
                    )
            # Hash stages interleaved across the block.
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    factor = 1 + (1 << val3)
                    v_factor = self.scratch_vconst(factor)
                    v_val1 = self.scratch_vconst(val1)
                    for bi, vec_i in enumerate(block_vecs):
                        v_val = val_cache + vec_i
                        slots.append(
                            ("valu", ("multiply_add", v_val, v_val, v_factor, v_val1))
                        )
                else:
                    v_val1 = self.scratch_vconst(val1)
                    v_val3 = self.scratch_vconst(val3)
                    for bi, vec_i in enumerate(block_vecs):
                        v_val = val_cache + vec_i
                        slots.append(
                            (
                                "valu",
                                (op1, v_tmp1_block + bi * VLEN, v_val, v_val1),
                            )
                        )
                    for bi, vec_i in enumerate(block_vecs):
                        v_val = val_cache + vec_i
                        slots.append(
                            (
                                "valu",
                                (op3, v_tmp2_block + bi * VLEN, v_val, v_val3),
                            )
                        )
                    for bi, vec_i in enumerate(block_vecs):
                        v_val = val_cache + vec_i
                        slots.append(
                            (
                                "valu",
                                (op2, v_val, v_tmp1_block + bi * VLEN, v_tmp2_block + bi * VLEN),
                            )
                        )
            # Index update.
            if fast_wrap and wrap_round:
                for _bi, vec_i in enumerate(block_vecs):
                    v_idx = idx_cache + vec_i
                    slots.append(("valu", ("+", v_idx, v_zero, v_zero)))
            else:
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(
                        ("valu", ("&", v_tmp3_block + bi * VLEN, v_val, v_one))
                    )
                for bi, _vec_i in enumerate(block_vecs):
                    slots.append(
                        (
                            "valu",
                            ("+", v_tmp3_block + bi * VLEN, v_tmp3_block + bi * VLEN, v_one),
                        )
                    )
                for bi, vec_i in enumerate(block_vecs):
                    v_idx = idx_cache + vec_i
                    slots.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                v_idx,
                                v_idx,
                                v_two,
                                v_tmp3_block + bi * VLEN,
                            ),
                        )
                    )
                if not fast_wrap:
                    for bi, vec_i in enumerate(block_vecs):
                        v_idx = idx_cache + vec_i
                        slots.append(
                            ("valu", ("<", v_tmp1_block + bi * VLEN, v_idx, v_n_nodes))
                        )
                        slots.append(
                            (
                                "flow",
                                ("vselect", v_idx, v_tmp1_block + bi * VLEN, v_idx, v_zero),
                            )
                        )
            return slots

        def vec_block_hash_slots(
            block_vecs,
            round_idx,
            wrap_round,
            node_const=None,
            node_pair=None,
            node_arith=None,
            node_prefetch=None,
        ):
            """Combined load + hash (non-pipelined fallback)."""
            slots = vec_block_load_slots(
                block_vecs, 0, node_const, node_pair, node_arith, node_prefetch
            )
            slots.extend(
                vec_block_hash_only_slots(
                    block_vecs,
                    0,
                    wrap_round,
                    node_const,
                    node_pair,
                    node_arith,
                    node_prefetch,
                )
            )
            return slots

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        if self.enable_debug:
            self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        wrap_period = forest_height + 1
        # Fast wrap relies on inputs starting at index 0 (Input.generate does this).
        fast_wrap = self.assume_zero_indices and not self.enable_debug
        root_node_val = None
        v_root_node = None
        v_level1_right = None
        v_level1_diff = None
        level_arith = {}
        # Enable level-1 parity shortcut: node_pair now uses multiply_add.
        enable_level1 = True
        if fast_wrap:
            root_node_val = self.alloc_scratch("root_node_val")
            v_root_node = self.alloc_scratch("v_root_node", VLEN)
            self.add("load", ("load", root_node_val, self.scratch["forest_values_p"]))
            self.add("valu", ("vbroadcast", v_root_node, root_node_val))
            if enable_level1 and rounds > 1:
                level1_addr = self.alloc_scratch("level1_addr")
                level1_left = self.alloc_scratch("level1_left")
                level1_right = self.alloc_scratch("level1_right")
                v_level1_left = self.alloc_scratch("v_level1_left", VLEN)
                v_level1_right = self.alloc_scratch("v_level1_right", VLEN)
                v_level1_diff = self.alloc_scratch("v_level1_diff", VLEN)
                self.add(
                    "alu",
                    (
                        "+",
                        level1_addr,
                        self.scratch["forest_values_p"],
                        self.scratch_const(1),
                    ),
                )
                self.add("load", ("load", level1_left, level1_addr))
                self.add("flow", ("add_imm", level1_addr, level1_addr, 1))
                self.add("load", ("load", level1_right, level1_addr))
                self.add("valu", ("vbroadcast", v_level1_left, level1_left))
                self.add("valu", ("vbroadcast", v_level1_right, level1_right))
                self.add("valu", ("-", v_level1_diff, v_level1_left, v_level1_right))
            if enable_arith:
                max_arith = min(self.max_arith_level, forest_height, 4)
                for level in range(2, max_arith + 1):
                    level_size = 1 << level
                    level_start = level_size - 1
                    level_addr = self.alloc_scratch(f"level{level}_addr")
                    self.add(
                        "alu",
                        (
                            "+",
                            level_addr,
                            self.scratch["forest_values_p"],
                            self.scratch_const(level_start),
                        ),
                    )
                    v_level_bases = []
                    v_level_diffs = []
                    for pair in range(level_size // 2):
                        even_val = tmp1
                        odd_val = tmp2
                        diff_val = tmp3
                        self.add("load", ("load", even_val, level_addr))
                        self.add("flow", ("add_imm", level_addr, level_addr, 1))
                        self.add("load", ("load", odd_val, level_addr))
                        if pair < (level_size // 2) - 1:
                            self.add("flow", ("add_imm", level_addr, level_addr, 1))
                        # base is even-indexed value; diff=odd-even makes bit=1 select odd.
                        self.add("alu", ("-", diff_val, odd_val, even_val))
                        v_base = self.alloc_scratch(
                            f"v_level{level}_base_{2 * pair}", VLEN
                        )
                        v_diff = self.alloc_scratch(
                            f"v_level{level}_diff_{2 * pair}", VLEN
                        )
                        self.add("valu", ("vbroadcast", v_base, even_val))
                        self.add("valu", ("vbroadcast", v_diff, diff_val))
                        v_level_bases.append(v_base)
                        v_level_diffs.append(v_diff)
                    level_arith[level] = {
                        "level": level,
                        "start": level_start,
                        "v_start": self.scratch_vconst(level_start),
                        "bases": v_level_bases,
                        "diffs": v_level_diffs,
                    }
        vec_count = vec_end // VLEN
        # Cross-round pipelining for better load utilization
        block_limit = vec_count - (vec_count % block_size)
        num_blocks = block_limit // block_size
        use_cross_round = (
            not self.enable_debug
            and vec_count >= block_size
            and num_blocks >= 2
            and not use_special
            and block_limit == vec_count  # no remainder
        )
        v_node_prefetch = None
        # Prefetch relies on arith rounds and the cross-round pipeline to create room for lookahead loads.
        enable_prefetch = (
            self.enable_prefetch and enable_arith and rounds > 1 and use_cross_round
        )
        if enable_prefetch:
            v_node_prefetch = self.alloc_scratch("v_node_prefetch", vec_count * VLEN)

        round_info = []
        prefetch_active = [False] * rounds
        prefetch_next = [False] * rounds
        if use_cross_round:
            for round in range(rounds):
                level = round % wrap_period
                uniform_round = fast_wrap and level == 0
                binary_round = fast_wrap and level == 1
                arith_round = fast_wrap and level in level_arith
                node_const = v_root_node if uniform_round else None
                node_pair = (
                    (v_level1_right, v_level1_diff)
                    if binary_round and v_level1_diff
                    else None
                )
                node_arith = level_arith[level] if arith_round else None
                round_info.append(
                    {
                        "round": round,
                        "level": level,
                        "wrap_round": level == forest_height,
                        "node_const": node_const,
                        "node_pair": node_pair,
                        "node_arith": node_arith,
                        "load_needed": (
                            node_const is None
                            and node_pair is None
                            and node_arith is None
                        ),
                        "arith_round": arith_round,
                    }
                )
            if v_node_prefetch is not None:
                # Prefetch the next round's node values when an arith round appears.
                for round in range(rounds - 1):
                    info = round_info[round]
                    next_info = round_info[round + 1]
                    if info["arith_round"] and next_info["load_needed"]:
                        prefetch_next[round] = True
                        prefetch_active[round + 1] = True

        if use_cross_round:
            block_0_vecs = [offset * VLEN for offset in range(block_size)]
            last_block_idx = num_blocks - 1
            last_start = last_block_idx * block_size
            last_block_vecs = [
                (last_start + offset) * VLEN for offset in range(block_size)
            ]
            all_block_vecs = [
                [(b * block_size + offset) * VLEN for offset in range(block_size)]
                for b in range(num_blocks)
            ]

            pending_prev = False  # Tracks deferred epilogue so we don't double-hash a round.
            prev_info = None
            prev_use_prefetch = False

            for round in range(rounds):
                info = round_info[round]
                use_prefetch = prefetch_active[round]
                do_prefetch_next = prefetch_next[round]
                node_prefetch = v_node_prefetch if use_prefetch else None
                special_round = info["arith_round"] and enable_prefetch

                if pending_prev:
                    last_buf = last_block_idx % 2
                    prev_node_prefetch = v_node_prefetch if prev_use_prefetch else None
                    hash_prev = vec_block_hash_only_slots(
                        last_block_vecs,
                        last_buf,
                        prev_info["wrap_round"],
                        prev_info["node_const"],
                        prev_info["node_pair"],
                        prev_info["node_arith"],
                        prev_node_prefetch,
                    )
                    if special_round:
                        body.extend(hash_prev)
                    else:
                        load_slots = vec_block_load_slots(
                            block_0_vecs,
                            0,
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                            node_prefetch,
                        )
                        body.extend(interleave_slots(hash_prev, load_slots))

                if special_round:
                    body.extend(
                        vec_block_hash_only_slots(
                            block_0_vecs,
                            0,
                            info["wrap_round"],
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                            node_prefetch,
                        )
                    )
                    for block_idx in range(1, num_blocks):
                        buf_idx = block_idx % 2
                        hash_slots = vec_block_hash_only_slots(
                            all_block_vecs[block_idx],
                            buf_idx,
                            info["wrap_round"],
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                            node_prefetch,
                        )
                        # Prefetch only block 0, issued one block behind so idx updates are committed.
                        load_slots = []
                        if do_prefetch_next and block_idx == 1:
                            load_slots = vec_block_prefetch_slots(
                                all_block_vecs[0], v_node_prefetch
                            )
                        body.extend(interleave_slots(hash_slots, load_slots))
                    pending_prev = False
                else:
                    start_block = 1
                    if use_prefetch:
                        # Block 0 is prefetched; hash it while loading block 1.
                        hash_slots = vec_block_hash_only_slots(
                            block_0_vecs,
                            0,
                            info["wrap_round"],
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                            node_prefetch,
                        )
                        load_slots = vec_block_load_slots(
                            all_block_vecs[1],
                            1,
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                        )
                        body.extend(interleave_slots(hash_slots, load_slots))
                        start_block = 2
                    elif not pending_prev:
                        body.extend(
                            vec_block_load_slots(
                                block_0_vecs,
                                0,
                                info["node_const"],
                                info["node_pair"],
                                info["node_arith"],
                            )
                        )
                    for block_idx in range(start_block, num_blocks):
                        prev_buf = (block_idx - 1) % 2
                        curr_buf = block_idx % 2
                        hash_slots = vec_block_hash_only_slots(
                            all_block_vecs[block_idx - 1],
                            prev_buf,
                            info["wrap_round"],
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                        )
                        load_slots = vec_block_load_slots(
                            all_block_vecs[block_idx],
                            curr_buf,
                            info["node_const"],
                            info["node_pair"],
                            info["node_arith"],
                        )
                        body.extend(interleave_slots(hash_slots, load_slots))
                    pending_prev = True

                prev_info = info
                # Prefetch only covers block 0; epilogue uses normal buffers.
                prev_use_prefetch = False

            if pending_prev:
                last_buf = last_block_idx % 2
                node_prefetch = v_node_prefetch if prev_use_prefetch else None
                body.extend(
                    vec_block_hash_only_slots(
                        last_block_vecs,
                        last_buf,
                        prev_info["wrap_round"],
                        prev_info["node_const"],
                        prev_info["node_pair"],
                        prev_info["node_arith"],
                        node_prefetch,
                    )
                )

        else:
            # Fallback: per-round processing
            for round in range(rounds):
                wrap_round = (round % wrap_period) == forest_height
                level = round % wrap_period
                uniform_round = fast_wrap and (round % wrap_period) == 0
                node_const = v_root_node if uniform_round else None
                binary_round = fast_wrap and (round % wrap_period) == 1
                node_pair = (
                    (v_level1_right, v_level1_diff)
                    if binary_round and v_level1_diff
                    else None
                )
                arith_round = fast_wrap and level in level_arith
                node_arith = level_arith[level] if arith_round else None
                special_round = (
                    use_special
                    and level <= max_special
                    and not uniform_round
                    and not binary_round
                    and not arith_round
                )
                if vec_count > 0:
                    if special_round:
                        for vec in range(vec_count):
                            buf = vec % 2
                            body.extend(vec_special_slots(vec * VLEN, buf, level))
                            body.extend(vec_hash_slots(vec * VLEN, buf, round, wrap_round))
                    elif not self.enable_debug and vec_count >= block_size:
                        block_limit = vec_count - (vec_count % block_size)
                        num_blocks = block_limit // block_size
                        if num_blocks >= 2:
                            block_0_vecs = [offset * VLEN for offset in range(block_size)]
                            all_block_vecs = [
                                [(b * block_size + offset) * VLEN for offset in range(block_size)]
                                for b in range(num_blocks)
                            ]
                            body.extend(
                                vec_block_load_slots(
                                    block_0_vecs, 0, node_const, node_pair, node_arith
                                )
                            )
                            for block_idx in range(1, num_blocks):
                                prev_buf = (block_idx - 1) % 2
                                curr_buf = block_idx % 2
                                hash_slots = vec_block_hash_only_slots(
                                    all_block_vecs[block_idx - 1],
                                    prev_buf,
                                    wrap_round,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                                load_slots = vec_block_load_slots(
                                    all_block_vecs[block_idx],
                                    curr_buf,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                                body.extend(interleave_slots(hash_slots, load_slots))
                            last_buf = (num_blocks - 1) % 2
                            body.extend(
                                vec_block_hash_only_slots(
                                    all_block_vecs[num_blocks - 1],
                                    last_buf,
                                    wrap_round,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                            )
                        else:
                            for vec in range(0, block_limit, block_size):
                                block_vecs = [(vec + offset) * VLEN for offset in range(block_size)]
                                body.extend(
                                    vec_block_hash_slots(
                                        block_vecs,
                                        round,
                                        wrap_round,
                                        node_const,
                                        node_pair,
                                        node_arith,
                                    )
                                )
                        if block_limit < vec_count:
                            start_vec = block_limit
                            body.extend(
                                vec_load_slots(
                                    start_vec * VLEN,
                                    start_vec % 2,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                            )
                            for vec in range(start_vec + 1, vec_count):
                                prev = vec - 1
                                hash_slots = vec_hash_slots(
                                    prev * VLEN,
                                    prev % 2,
                                    round,
                                    wrap_round,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                                load_slots = vec_load_slots(
                                    vec * VLEN,
                                    vec % 2,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                                body.extend(interleave_slots(hash_slots, load_slots))
                            last_vec = vec_count - 1
                            body.extend(
                                vec_hash_slots(
                                    last_vec * VLEN,
                                    last_vec % 2,
                                    round,
                                    wrap_round,
                                    node_const,
                                    node_pair,
                                    node_arith,
                                )
                            )
                    else:
                        body.extend(
                            vec_load_slots(0, 0, node_const, node_pair, node_arith)
                        )
                        for vec in range(1, vec_count):
                            prev = vec - 1
                            hash_slots = vec_hash_slots(
                                prev * VLEN,
                                prev % 2,
                                round,
                                wrap_round,
                                node_const,
                                node_pair,
                                node_arith,
                            )
                            load_slots = vec_load_slots(
                                vec * VLEN,
                                vec % 2,
                                node_const,
                                node_pair,
                                node_arith,
                            )
                            body.extend(interleave_slots(hash_slots, load_slots))
                        last_vec = vec_count - 1
                        body.extend(
                            vec_hash_slots(
                                last_vec * VLEN,
                                last_vec % 2,
                                round,
                                wrap_round,
                                node_const,
                                node_pair,
                                node_arith,
                            )
                        )

        if vec_end < batch_size:
            for round in range(rounds):
                wrap_round = (round % wrap_period) == forest_height
                for i in range(vec_end, batch_size):
                    idx_addr = idx_cache + i
                    val_addr = val_cache + i
                    # idx = cached index
                    if self.enable_debug:
                        body.append(("debug", ("compare", idx_addr, (round, i, "idx"))))
                    # val = cached value
                    if self.enable_debug:
                        body.append(("debug", ("compare", val_addr, (round, i, "val"))))
                    # node_val = mem[forest_values_p + idx]
                    body.append(
                        ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_addr))
                    )
                    body.append(("load", ("load", tmp_node_val, tmp_addr)))
                    if self.enable_debug:
                        body.append(
                            ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                        )
                    # val = myhash(val ^ node_val)
                    body.append(("alu", ("^", val_addr, val_addr, tmp_node_val)))
                    body.extend(self.build_hash(val_addr, tmp1, tmp2, round, i))
                    if self.enable_debug:
                        body.append(
                            ("debug", ("compare", val_addr, (round, i, "hashed_val")))
                        )
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    if fast_wrap:
                        if wrap_round:
                            body.append(("alu", ("+", idx_addr, zero_const, zero_const)))
                        else:
                            body.append(("alu", ("&", tmp3, val_addr, one_const)))
                            body.append(("alu", ("+", tmp3, tmp3, one_const)))
                            body.append(("alu", ("*", idx_addr, idx_addr, two_const)))
                            body.append(("alu", ("+", idx_addr, idx_addr, tmp3)))
                    else:
                        body.append(("alu", ("&", tmp3, val_addr, one_const)))
                        body.append(("alu", ("+", tmp3, tmp3, one_const)))
                        body.append(("alu", ("*", idx_addr, idx_addr, two_const)))
                        body.append(("alu", ("+", idx_addr, idx_addr, tmp3)))
                    if self.enable_debug:
                        body.append(
                            ("debug", ("compare", idx_addr, (round, i, "next_idx")))
                        )
                    if not fast_wrap:
                        body.append(
                            ("alu", ("<", tmp1, idx_addr, self.scratch["n_nodes"]))
                        )
                        body.append(
                            ("flow", ("select", idx_addr, tmp1, idx_addr, zero_const))
                        )
                    if self.enable_debug:
                        body.append(
                            ("debug", ("compare", idx_addr, (round, i, "wrapped_idx")))
                        )

        # Write back cached values; indices are optional for benchmark speed.
        val_store_ptr = self.alloc_scratch("val_store_ptr")
        if write_indices:
            idx_store_ptr = self.alloc_scratch("idx_store_ptr")
            body.append(
                ("alu", ("+", idx_store_ptr, self.scratch["inp_indices_p"], zero_const))
            )
        body.append(
            ("alu", ("+", val_store_ptr, self.scratch["inp_values_p"], zero_const))
        )
        for i in range(0, vec_end, VLEN):
            if write_indices:
                body.append(("store", ("vstore", idx_store_ptr, idx_cache + i)))
            body.append(("store", ("vstore", val_store_ptr, val_cache + i)))
            if write_indices:
                body.append(("flow", ("add_imm", idx_store_ptr, idx_store_ptr, VLEN)))
            body.append(("flow", ("add_imm", val_store_ptr, val_store_ptr, VLEN)))
        for i in range(vec_end, batch_size):
            if write_indices:
                body.append(("store", ("store", idx_store_ptr, idx_cache + i)))
            body.append(("store", ("store", val_store_ptr, val_cache + i)))
            if write_indices:
                body.append(("flow", ("add_imm", idx_store_ptr, idx_store_ptr, 1)))
            body.append(("flow", ("add_imm", val_store_ptr, val_store_ptr, 1)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    enable_debug: bool | None = None,
    assume_zero_indices: bool | None = None,
    max_special_level: int | None = None,
    max_arith_level: int | None = None,
    enable_prefetch: bool | None = None,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    if enable_debug is None:
        enable_debug = trace
    if assume_zero_indices is None:
        assume_zero_indices = True
    if max_special_level is None:
        max_special_level = -1
    if max_arith_level is None:
        max_arith_level = -1
    if enable_prefetch is None:
        enable_prefetch = False
    kb = KernelBuilder(
        enable_debug=enable_debug,
        assume_zero_indices=assume_zero_indices,
        max_special_level=max_special_level,
        max_arith_level=max_arith_level,
        enable_prefetch=enable_prefetch,
    )
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
