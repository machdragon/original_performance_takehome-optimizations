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

        def find_pullable_load(start_idx, future_writes, future_reads, lookahead=16):
            skipped_writes = set()
            skipped_reads = set()
            for j in range(start_idx, min(len(slots_info), start_idx + lookahead)):
                info = slots_info[j]
                if info is None:
                    continue
                if info["barrier"]:
                    break
                reads = info["reads"]
                writes = info["writes"]
                if info["engine"] == "load":
                    if reads & skipped_writes:
                        pass
                    elif reads & future_writes:
                        pass
                    elif writes & future_writes:
                        pass
                    elif writes & future_reads:
                        pass
                    elif writes & skipped_writes:
                        pass
                    elif writes & skipped_reads:
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
                future_writes = bundle_writes | writes
                future_reads = bundle_reads | reads
                pull_idx = find_pullable_load(i + 1, future_writes, future_reads)
                if pull_idx is not None:
                    pull = slots_info[pull_idx]
                    slots_info[pull_idx] = None
                    bundle.setdefault("load", []).append(pull["slot"])
                    bundle_reads.update(pull["reads"])
                    bundle_writes.update(pull["writes"])

            if len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]:
                flush()
                continue
            if reads & bundle_writes:
                flush()
                continue
            if writes & bundle_writes:
                flush()
                continue

            bundle.setdefault(engine, []).append(slot)
            bundle_reads.update(reads)
            bundle_writes.update(writes)
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
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation that uses SIMD for the batch dimension.
        """
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
        idx_load_ptr = self.alloc_scratch("idx_load_ptr")
        val_load_ptr = self.alloc_scratch("val_load_ptr")
        self.add("alu", ("+", idx_load_ptr, self.scratch["inp_indices_p"], zero_const))
        self.add("alu", ("+", val_load_ptr, self.scratch["inp_values_p"], zero_const))
        vec_end = batch_size - (batch_size % VLEN)
        for i in range(0, vec_end, VLEN):
            self.add("load", ("vload", idx_cache + i, idx_load_ptr))
            self.add("load", ("vload", val_cache + i, val_load_ptr))
            self.add("flow", ("add_imm", idx_load_ptr, idx_load_ptr, VLEN))
            self.add("flow", ("add_imm", val_load_ptr, val_load_ptr, VLEN))
        for i in range(vec_end, batch_size):
            self.add("load", ("load", idx_cache + i, idx_load_ptr))
            self.add("load", ("load", val_cache + i, val_load_ptr))
            self.add("flow", ("add_imm", idx_load_ptr, idx_load_ptr, 1))
            self.add("flow", ("add_imm", val_load_ptr, val_load_ptr, 1))

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
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_values_p", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))
        for _, val1, _, _, val3 in HASH_STAGES:
            self.scratch_vconst(val1)
            self.scratch_vconst(val3)

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
            if not load_slots:
                return hash_slots
            if not hash_slots:
                return load_slots
            combined = []
            h_len = len(hash_slots)
            l_len = len(load_slots)
            spacing = max(1, h_len // (l_len + 1))
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
                    since_load = 0
                elif h_i >= h_len and l_i < l_len:
                    combined.append(load_slots[l_i])
                    l_i += 1
            return combined

        def vec_load_slots(vec_i, buf_idx):
            slots = []
            v_idx = idx_cache + vec_i
            v_addr_buf = v_addr[buf_idx]
            v_node_buf = v_node_val[buf_idx]
            slots.append(("valu", ("+", v_addr_buf, v_idx, v_forest_p)))
            for lane in range(VLEN):
                slots.append(("load", ("load_offset", v_node_buf, v_addr_buf, lane)))
            return slots

        def vec_hash_slots(vec_i, buf_idx, round_idx, wrap_round):
            slots = []
            v_idx = idx_cache + vec_i
            v_val = val_cache + vec_i
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
                    slots.append(("valu", ("+", v_idx, v_idx, v_idx)))
                    slots.append(("valu", ("+", v_idx, v_idx, v_tmp3)))
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
        vec_count = vec_end // VLEN
        for round in range(rounds):
            wrap_round = (round % wrap_period) == forest_height
            level = round % wrap_period
            special_round = use_special and level <= max_special
            if vec_count > 0:
                if special_round:
                    for vec in range(vec_count):
                        buf = vec % 2
                        body.extend(vec_special_slots(vec * VLEN, buf, level))
                        body.extend(vec_hash_slots(vec * VLEN, buf, round, wrap_round))
                else:
                    # Prologue: load vector 0
                    body.extend(vec_load_slots(0, 0))
                    for vec in range(1, vec_count):
                        prev = vec - 1
                        hash_slots = vec_hash_slots(
                            prev * VLEN, prev % 2, round, wrap_round
                        )
                        load_slots = vec_load_slots(vec * VLEN, vec % 2)
                        body.extend(interleave_slots(hash_slots, load_slots))
                    # Epilogue: hash last vector
                    last_vec = vec_count - 1
                    body.extend(
                        vec_hash_slots(last_vec * VLEN, last_vec % 2, round, wrap_round)
                    )

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

        # Write back cached values and indices.
        idx_store_ptr = self.alloc_scratch("idx_store_ptr")
        val_store_ptr = self.alloc_scratch("val_store_ptr")
        body.append(
            ("alu", ("+", idx_store_ptr, self.scratch["inp_indices_p"], zero_const))
        )
        body.append(
            ("alu", ("+", val_store_ptr, self.scratch["inp_values_p"], zero_const))
        )
        for i in range(0, vec_end, VLEN):
            body.append(("store", ("vstore", idx_store_ptr, idx_cache + i)))
            body.append(("store", ("vstore", val_store_ptr, val_cache + i)))
            body.append(("flow", ("add_imm", idx_store_ptr, idx_store_ptr, VLEN)))
            body.append(("flow", ("add_imm", val_store_ptr, val_store_ptr, VLEN)))
        for i in range(vec_end, batch_size):
            body.append(("store", ("store", idx_store_ptr, idx_cache + i)))
            body.append(("store", ("store", val_store_ptr, val_cache + i)))
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
    kb = KernelBuilder(
        enable_debug=enable_debug,
        assume_zero_indices=assume_zero_indices,
        max_special_level=max_special_level,
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
