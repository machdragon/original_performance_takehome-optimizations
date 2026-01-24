
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
        enable_prefetch: bool = True,
        enable_level2_where: bool = True,
        enable_level2_valu: bool = False,
        enable_two_round_fusion: bool = False,
        enable_level3_where: bool = False,
        enable_level4_valu: bool = False,
        enable_unroll_8: bool = True,
        lookahead: int = 1024,
        block_size: int = 16,
        enable_second_pass: bool = False,
        enable_latency_aware: bool = False,
        enable_combining: bool = False,
        enable_software_pipeline: bool = False,
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
        self.enable_level2_where = enable_level2_where
        self.enable_level2_valu = enable_level2_valu
        self.enable_two_round_fusion = enable_two_round_fusion
        self.enable_level3_where = enable_level3_where
        self.enable_level4_valu = enable_level4_valu
        self.enable_unroll_8 = enable_unroll_8
        self.lookahead = lookahead
        self.block_size = block_size
        self.enable_second_pass = enable_second_pass
        self.enable_latency_aware = enable_latency_aware
        self.enable_combining = enable_combining
        self.enable_software_pipeline = enable_software_pipeline

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

        def can_add_now(info, recent_loads=None):
            if recent_loads is None:
                recent_loads = set()
            if engine_full(info["engine"]):
                return False
            if info["reads"] & bundle_writes:
                return False
            if info["writes"] & bundle_writes:
                return False

            if self.enable_latency_aware and info["engine"] in ["alu", "valu"]:
                if info["reads"] & recent_loads:
                    return False
            return True

        def find_pullable_slot(
            start_idx,
            future_writes,
            engine_filter=None,
            lookahead=None,
            init_skipped_reads=None,
            init_skipped_writes=None,
        ):
            if lookahead is None:
                lookahead = self.lookahead
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

            recent_loads = set()
            if self.enable_latency_aware and "load" in bundle:
                recent_loads = bundle_writes

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

            if engine == "load" and bundle_loads() >= SLOT_LIMITS["load"]:

                while not engine_full("valu") and bundle_loads() >= SLOT_LIMITS["load"]:
                    future_writes = bundle_writes | writes
                    pull_idx = find_pullable_slot(
                        i + 1,
                        future_writes,
                        engine_filter="valu",
                    )
                    if pull_idx is None:
                        break
                    pull = slots_info[pull_idx]
                    slots_info[pull_idx] = None
                    add_to_bundle(pull)

            if not can_add_now(info, recent_loads):
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

        if False and self.enable_combining:
            combined = []
            i = 0
            while i < len(instrs):
                current = instrs[i].copy()

                current_slots = sum(len(slots) for slots in current.values())
                max_slots = sum(SLOT_LIMITS.get(engine, 0) for engine in ["alu", "valu", "load", "store", "flow"])
                utilization = current_slots / max_slots if max_slots > 0 else 1.0

                if utilization < 0.8:
                    j = i + 1
                    merged_count = 0
                    while j < len(instrs) and merged_count < 2:
                        next_bundle = instrs[j].copy()

                        current_writes = set()
                        current_reads = set()
                        for engine, slots in current.items():
                            for slot in slots:
                                reads, writes, _ = self._slot_reads_writes(engine, slot)
                                current_reads.update(reads)
                                current_writes.update(writes)

                        next_reads = set()
                        next_writes = set()
                        for engine, slots in next_bundle.items():
                            for slot in slots:
                                reads, writes, _ = self._slot_reads_writes(engine, slot)
                                next_reads.update(reads)
                                next_writes.update(writes)

                        has_raw = bool(next_reads & current_writes)
                        has_waw = bool(next_writes & current_writes)
                        has_war = bool(next_writes & current_reads)

                        if has_raw or has_waw or has_war:
                            break

                        can_merge = True
                        merged = current.copy()
                        for engine, slots in next_bundle.items():
                            current_count = len(merged.get(engine, []))
                            next_count = len(slots)
                            limit = SLOT_LIMITS.get(engine, 64)
                            if current_count + next_count > limit:
                                can_merge = False
                                break
                            merged.setdefault(engine, []).extend(slots)

                        if can_merge:
                            current = merged
                            j += 1
                            merged_count += 1
                        else:
                            break
                    i = j if j > i + 1 else i + 1
                else:
                    i += 1

                combined.append(current)

            instrs = combined

        if self.enable_second_pass:
            for i in range(len(instrs)):
                bundle = instrs[i].copy()

                load_slots = bundle.pop("load", [])
                valu_slots = bundle.pop("valu", [])
                alu_slots = bundle.pop("alu", [])
                store_slots = bundle.pop("store", [])
                flow_slots = bundle.pop("flow", [])
                debug_slots = bundle.pop("debug", [])

                reordered = {}
                if load_slots:
                    reordered["load"] = load_slots[:SLOT_LIMITS["load"]]
                if valu_slots:
                    reordered["valu"] = valu_slots[:SLOT_LIMITS["valu"]]
                if alu_slots:
                    reordered["alu"] = alu_slots[:SLOT_LIMITS["alu"]]
                if store_slots:
                    reordered["store"] = store_slots[:SLOT_LIMITS["store"]]
                if flow_slots:
                    reordered["flow"] = flow_slots[:SLOT_LIMITS["flow"]]
                if debug_slots:
                    reordered["debug"] = debug_slots

                for engine, slots in bundle.items():
                    if slots:
                        reordered.setdefault(engine, []).extend(slots)

                instrs[i] = reordered

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        if self.scratch_ptr > SCRATCH_SIZE:
            usage_info = f"Scratch usage: {self.scratch_ptr}/{SCRATCH_SIZE} words"
            if name:
                usage_info += f" (failed allocating {length} words for '{name}')"
            raise AssertionError(f"Out of scratch space: {usage_info}")
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

    def build_vselect_tree(
        self,
        all_level_vecs: int,
        relative_idx_vec: int,
        level_size: int,
        temp_base: int,
        round: int,
        vec: int,
    ):
        slots = []

        if level_size <= 1:

            out = all_level_vecs
            if self.enable_debug:
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            out,
                            [(round, vec * VLEN + l, "node_val") for l in range(VLEN)],
                        ),
                    )
                )
            return out, slots

        current_out = all_level_vecs
        rel_idx = relative_idx_vec
        v_zero = self.scratch_vconst(0)

        mid1 = level_size // 2
        v_mid1 = self.scratch_vconst(mid1)
        go_left1 = self.alloc_scratch(f"go_left1_{vec}", VLEN)
        slots.append(("valu", ("<", go_left1, rel_idx, v_mid1)))

        left1 = current_out
        right1 = current_out + mid1 * VLEN
        out1 = temp_base
        slots.append(("flow", ("vselect", out1, go_left1, left1, right1)))

        subtract_mid1 = self.alloc_scratch(f"subtract_mid1_{vec}", VLEN)
        slots.append(("flow", ("vselect", subtract_mid1, go_left1, v_zero, v_mid1)))
        next_rel_idx1 = self.alloc_scratch(f"next_rel_idx1_{vec}", VLEN)
        slots.append(("valu", ("-", next_rel_idx1, rel_idx, subtract_mid1)))

        if level_size > 2:
            mid2 = mid1 // 2
            v_mid2 = self.scratch_vconst(mid2)
            go_left2 = self.alloc_scratch(f"go_left2_{vec}", VLEN)
            slots.append(("valu", ("<", go_left2, next_rel_idx1, v_mid2)))

            left2 = out1
            right2 = out1 + mid2 * VLEN
            out2 = temp_base + VLEN
            slots.append(("flow", ("vselect", out2, go_left2, left2, right2)))

            if level_size > 4:
                subtract_mid2 = self.alloc_scratch(f"subtract_mid2_{vec}", VLEN)
                slots.append(("flow", ("vselect", subtract_mid2, go_left2, v_zero, v_mid2)))
                next_rel_idx2 = self.alloc_scratch(f"next_rel_idx2_{vec}", VLEN)
                slots.append(("valu", ("-", next_rel_idx2, next_rel_idx1, subtract_mid2)))

                mid3 = mid2 // 2
                v_mid3 = self.scratch_vconst(mid3)
                go_left3 = self.alloc_scratch(f"go_left3_{vec}", VLEN)
                slots.append(("valu", ("<", go_left3, next_rel_idx2, v_mid3)))

                left3 = out2
                right3 = out2 + mid3 * VLEN
                out3 = temp_base + 2 * VLEN
                slots.append(("flow", ("vselect", out3, go_left3, left3, right3)))

                if level_size > 8:
                    subtract_mid3 = self.alloc_scratch(f"subtract_mid3_{vec}", VLEN)
                    slots.append(("flow", ("vselect", subtract_mid3, go_left3, v_zero, v_mid3)))
                    next_rel_idx3 = self.alloc_scratch(f"next_rel_idx3_{vec}", VLEN)
                    slots.append(("valu", ("-", next_rel_idx3, next_rel_idx2, subtract_mid3)))

                    mid4 = mid3 // 2
                    v_mid4 = self.scratch_vconst(mid4)
                    go_left4 = self.alloc_scratch(f"go_left4_{vec}", VLEN)
                    slots.append(("valu", ("<", go_left4, next_rel_idx3, v_mid4)))

                    left4 = out3
                    right4 = out3 + mid4 * VLEN
                    out4 = temp_base + 3 * VLEN
                    slots.append(("flow", ("vselect", out4, go_left4, left4, right4)))
                    final_out = out4
                else:
                    final_out = out3
            else:
                final_out = out2
        else:
            final_out = out1

        if self.enable_debug:
            slots.append(
                (
                    "debug",
                    (
                        "vcompare",
                        final_out,
                        [(round, vec * VLEN + l, "node_val") for l in range(VLEN)],
                    ),
                )
            )

        return final_out, slots

    def build_vselect_tree_reuse(
        self,
        all_level_vecs: int,
        relative_idx_vec: int,
        level_size: int,
        temp_base: int,
        final_temp: int,
        round: int,
        vec: int,
    ):
        slots = []

        if level_size <= 1:
            out = all_level_vecs
            if self.enable_debug:
                slots.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            out,
                            [(round, vec * VLEN + l, "node_val") for l in range(VLEN)],
                        ),
                    )
                )
            return out, slots

        current_out = all_level_vecs
        rel_idx = relative_idx_vec
        v_zero = self.scratch_vconst(0)
        temp_offset = 0
        
        mid1 = level_size // 2
        v_mid1 = self.scratch_vconst(mid1)
        go_left1 = temp_base + temp_offset
        temp_offset += VLEN
        slots.append(("valu", ("<", go_left1, rel_idx, v_mid1)))

        left1 = current_out
        right1 = current_out + mid1 * VLEN
        out1 = temp_base + temp_offset
        temp_offset += VLEN
        slots.append(("flow", ("vselect", out1, go_left1, left1, right1)))

        subtract_mid1 = temp_base + temp_offset
        temp_offset += VLEN
        slots.append(("flow", ("vselect", subtract_mid1, go_left1, v_zero, v_mid1)))
        next_rel_idx1 = temp_base + temp_offset
        temp_offset += VLEN
        slots.append(("valu", ("-", next_rel_idx1, rel_idx, subtract_mid1)))

        if level_size > 2:
            mid2 = mid1 // 2
            v_mid2 = self.scratch_vconst(mid2)
            go_left2 = temp_base + temp_offset
            temp_offset += VLEN
            slots.append(("valu", ("<", go_left2, next_rel_idx1, v_mid2)))

            left2 = out1
            right2 = out1 + mid2 * VLEN
            out2 = temp_base + temp_offset
            temp_offset += VLEN
            slots.append(("flow", ("vselect", out2, go_left2, left2, right2)))


            if level_size > 4:

                subtract_mid2 = temp_base + temp_offset
                temp_offset += VLEN
                slots.append(("flow", ("vselect", subtract_mid2, go_left2, v_zero, v_mid2)))
                next_rel_idx2 = temp_base + temp_offset
                temp_offset += VLEN
                slots.append(("valu", ("-", next_rel_idx2, next_rel_idx1, subtract_mid2)))

                mid3 = mid2 // 2
                v_mid3 = self.scratch_vconst(mid3)
                go_left3 = temp_base + temp_offset
                temp_offset += VLEN
                slots.append(("valu", ("<", go_left3, next_rel_idx2, v_mid3)))

                left3 = out2
                right3 = out2 + mid3 * VLEN
                out3 = temp_base + temp_offset
                temp_offset += VLEN
                slots.append(("flow", ("vselect", out3, go_left3, left3, right3)))

                if level_size > 8:
                    subtract_mid3 = temp_base + temp_offset
                    temp_offset += VLEN
                    slots.append(("flow", ("vselect", subtract_mid3, go_left3, v_zero, v_mid3)))
                    next_rel_idx3 = temp_base + temp_offset
                    temp_offset += VLEN
                    slots.append(("valu", ("-", next_rel_idx3, next_rel_idx2, subtract_mid3)))

                    mid4 = mid3 // 2
                    v_mid4 = self.scratch_vconst(mid4)
                    go_left4 = temp_base + temp_offset
                    temp_offset += VLEN
                    slots.append(("valu", ("<", go_left4, next_rel_idx3, v_mid4)))

                    left4 = out3
                    right4 = out3 + mid4 * VLEN
                    out4 = final_temp
                    slots.append(("flow", ("vselect", out4, go_left4, left4, right4)))
                    final_out = out4
                else:
                    final_out = out3
            else:
                final_out = out2
        else:
            final_out = out1

        if self.enable_debug:
            slots.append(
                (
                    "debug",
                    (
                        "vcompare",
                        final_out,
                        [(round, vec * VLEN + l, "node_val") for l in range(VLEN)],
                    ),
                )
            )

        return final_out, slots

    def _interleave_slots(self, hash_slots, load_slots):
        if not load_slots:
            return hash_slots
        if not hash_slots:
            return load_slots
        combined = []
        h_len = len(hash_slots)
        l_len = len(load_slots)
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

    def build_vselect_tree_level2_unrolled(
        self, all_level_vecs, relative_idx_vec, temp_base, round, vec
    ):
        slots = []

        v_mid1 = self.scratch_vconst(2)
        go_left1 = self.alloc_scratch(f"go_left1_{vec}", VLEN)
        slots.append(("valu", ("<", go_left1, relative_idx_vec, v_mid1)))

        v_one = self.scratch_vconst(1)
        go_left2_left = self.alloc_scratch(f"go_left2_left_{vec}", VLEN)
        slots.append(("valu", ("<", go_left2_left, relative_idx_vec, v_one)))
        node0 = all_level_vecs
        node1 = all_level_vecs + VLEN
        out_left = temp_base
        slots.append(("flow", ("vselect", out_left, go_left2_left, node0, node1)))

        rel_idx_right = self.alloc_scratch(f"rel_idx_right_{vec}", VLEN)
        v_two = self.scratch_vconst(2)
        slots.append(("valu", ("-", rel_idx_right, relative_idx_vec, v_two)))
        go_left2_right = self.alloc_scratch(f"go_left2_right_{vec}", VLEN)
        slots.append(("valu", ("<", go_left2_right, rel_idx_right, v_one)))
        node2 = all_level_vecs + 2 * VLEN
        node3 = all_level_vecs + 3 * VLEN
        out_right = temp_base + VLEN
        slots.append(("flow", ("vselect", out_right, go_left2_right, node2, node3)))

        final_out = temp_base + 2 * VLEN
        slots.append(("flow", ("vselect", final_out, go_left1, out_left, out_right)))

        return final_out, slots

    def _vec_block_load_slots_specialized(
        self, block_vecs, buf_idx, info, v_root_node, v_level1_right, v_level1_diff,
        idx_cache, v_forest_p, v_addr, v_node_block, level2_base_addr_const, level2_vecs_base, level2_tree_tmp_base, level2_addr_temp, level2_scalars_temp, v_zero
    ):
        slots = []
        node_buf = v_node_block[buf_idx]

        if info["uniform_round"]:

            return slots
        elif info["binary_round"]:

            return slots
        elif info.get("level2_round", False):

            slots.append(("alu", ("+", level2_addr_temp, self.scratch["forest_values_p"], level2_base_addr_const)))
            for i in range(4):
                slots.append(("load", ("load", level2_scalars_temp + i, level2_addr_temp)))
                slots.append(("flow", ("add_imm", level2_addr_temp, level2_addr_temp, 1)))

            level2_vecs = level2_vecs_base
            for i in range(4):
                slots.append(("valu", ("vbroadcast", level2_vecs + i * VLEN, level2_scalars_temp + i)))

            for bi, vec_i in enumerate(block_vecs):
                v_idx = idx_cache + vec_i
                v_level_start = self.scratch_vconst(3)
                relative_idx_vec = self.alloc_scratch(f"rel_idx_l2_{bi}_{buf_idx}", VLEN)
                slots.append(("valu", ("-", relative_idx_vec, v_idx, v_level_start)))

                tree_temp = level2_vecs_base + 4 * VLEN
                final_out, tree_slots = self.build_vselect_tree(
                    level2_vecs, relative_idx_vec, 4, tree_temp, info["round"], vec_i // VLEN
                )
                slots.extend(tree_slots)

                slots.append(("valu", ("+", node_buf + bi * VLEN, final_out, v_zero)))
            return slots

        for bi, vec_i in enumerate(block_vecs):
            v_idx = idx_cache + vec_i
            slots.append(("valu", ("+", v_addr[0], v_idx, v_forest_p)))
            for lane in range(VLEN):
                slots.append(("load", ("load_offset", node_buf + bi * VLEN, v_addr[0], lane)))
        return slots

    def _vec_block_hash_only_slots_specialized(
        self, block_vecs, buf_idx, wrap_round, info, v_root_node, v_level1_right, v_level1_diff,
        idx_cache, val_cache, v_tmp1_block, v_tmp2_block, v_tmp3_block,
        v_zero, v_one, v_two, v_n_nodes, v_node_block
    ):
        slots = []
        node_buf = v_node_block[buf_idx]

        if info["uniform_round"]:
            for _bi, vec_i in enumerate(block_vecs):
                v_val = val_cache + vec_i
                slots.append(("valu", ("^", v_val, v_val, v_root_node)))
        elif info["binary_round"]:
            for bi, vec_i in enumerate(block_vecs):
                v_idx = idx_cache + vec_i
                slots.append(("valu", ("&", v_tmp1_block + bi * VLEN, v_idx, v_one)))
                slots.append(("valu", ("multiply_add", v_tmp3_block + bi * VLEN, v_tmp1_block + bi * VLEN, v_level1_diff, v_level1_right)))
            for bi, vec_i in enumerate(block_vecs):
                v_val = val_cache + vec_i
                slots.append(("valu", ("^", v_val, v_val, v_tmp3_block + bi * VLEN)))
        else:
            for bi, vec_i in enumerate(block_vecs):
                v_val = val_cache + vec_i
                slots.append(("valu", ("^", v_val, v_val, node_buf + bi * VLEN)))

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                v_factor = self.scratch_vconst(factor)
                v_val1 = self.scratch_vconst(val1)
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(("valu", ("multiply_add", v_val, v_val, v_factor, v_val1)))
            else:
                v_val1 = self.scratch_vconst(val1)
                v_val3 = self.scratch_vconst(val3)
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(("valu", (op1, v_tmp1_block + bi * VLEN, v_val, v_val1)))
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(("valu", (op3, v_tmp2_block + bi * VLEN, v_val, v_val3)))
                for bi, vec_i in enumerate(block_vecs):
                    v_val = val_cache + vec_i
                    slots.append(("valu", (op2, v_val, v_tmp1_block + bi * VLEN, v_tmp2_block + bi * VLEN)))

        if wrap_round:
            for _bi, vec_i in enumerate(block_vecs):
                v_idx = idx_cache + vec_i
                slots.append(("valu", ("+", v_idx, v_zero, v_zero)))
        else:
            for bi, vec_i in enumerate(block_vecs):
                v_val = val_cache + vec_i
                slots.append(("valu", ("&", v_tmp3_block + bi * VLEN, v_val, v_one)))
            for bi, _vec_i in enumerate(block_vecs):
                slots.append(("valu", ("+", v_tmp3_block + bi * VLEN, v_tmp3_block + bi * VLEN, v_one)))
            for bi, vec_i in enumerate(block_vecs):
                v_idx = idx_cache + vec_i
                slots.append(("valu", ("multiply_add", v_idx, v_idx, v_two, v_tmp3_block + bi * VLEN)))

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
        if write_indicies is not None:
            write_indices = write_indicies

        if (forest_height, rounds, batch_size) == (10, 16, 256):
            return self.build_kernel_10_16_256(n_nodes, write_indices)

        return self.build_kernel_general(
            forest_height, n_nodes, batch_size, rounds, write_indices
        )

    def build_kernel_10_16_256(self, n_nodes: int, write_indices: bool = True):

        return self.build_kernel_general(10, n_nodes, 256, 16, write_indices)

    def build_kernel_general(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        write_indices: bool = True,
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

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
            if v not in self.scratch:
                self.alloc_scratch(v, 1)
        # Skip loads if overfitting path already initialized
        if not getattr(self, '_overfit_init_done', False):
            for i, v in enumerate(init_vars):
                self.add("load", ("const", tmp1, i))
                self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        idx_cache = self.alloc_scratch("idx_cache", batch_size)
        val_cache = self.alloc_scratch("val_cache", batch_size)
        val_load_ptr = self.alloc_scratch("val_load_ptr")
        self.add("alu", ("+", val_load_ptr, self.scratch["inp_values_p"], zero_const))

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

        vec_count = vec_end // VLEN
        block_size = self.block_size
        block_limit = vec_count - (vec_count % block_size)
        num_blocks = block_limit // block_size
        max_special = min(self.max_special_level, forest_height)
        use_special = self.assume_zero_indices and max_special >= 0 and max_special < 4
        use_cross_round = (
            not self.enable_debug
            and vec_count >= block_size
            and num_blocks >= 2
            and not use_special
            and block_limit == vec_count
        )

        enable_arith = self.max_arith_level >= 2

        v_node_val_0 = self.alloc_scratch("v_node_val_0", VLEN)
        v_addr_0 = self.alloc_scratch("v_addr_0", VLEN)
        if use_cross_round:
            v_node_val = [v_node_val_0, v_node_val_0]
            v_addr = [v_addr_0, v_addr_0]
        else:
            v_node_val = [
                v_node_val_0,
                self.alloc_scratch("v_node_val_1", VLEN),
            ]
            v_addr = [
                v_addr_0,
                self.alloc_scratch("v_addr_1", VLEN),
            ]
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        if use_cross_round:
            v_tmp2 = v_tmp1
            v_tmp3 = v_tmp1
        else:
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

        v_node_block = [
            self.alloc_scratch("v_node_block_A", block_size * VLEN),
            self.alloc_scratch("v_node_block_B", block_size * VLEN),
        ]



        v_tmp1_block = self.alloc_scratch("v_tmp1_block", block_size * VLEN)
        v_tmp2_block = self.alloc_scratch("v_tmp2_block", block_size * VLEN)
        
        v_tmp4_block = None
        if enable_arith:
            v_tmp4_block = self.alloc_scratch("v_tmp4_block", block_size * VLEN)

        enable_level2_where = self.enable_level2_where
        enable_level3_where = self.enable_level3_where
        
        can_alias_tmp23 = not enable_level2_where and not enable_level3_where
        if can_alias_tmp23:
            v_tmp3_block = v_tmp2_block
        else:
            v_tmp3_block = self.alloc_scratch("v_tmp3_block", block_size * VLEN)
        level2_base_addr_const = self.scratch_const(3)
        level2_vecs_base = v_node_block[0]
        level4_vecs_base = v_node_block[0] if self.enable_level4_valu else None
        level4_diffs_base = v_node_block[1] if self.enable_level4_valu else None
        level2_tree_tmp_base = None
        level2_addr_temp = tmp1
        level2_scalars_base = v_tmp1

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

        upper_levels = {}
        precompute_max_level = 4
        enable_level4_precompute = False
        if enable_level4_precompute:
            level4_size = 16
            level4_start = 15
            level4_values = self.alloc_scratch("level4_values", level4_size)
            level4_vecs = self.alloc_scratch("level4_vecs", level4_size * VLEN)
            level4_addr = self.alloc_scratch("level4_addr")
            level_base = level4_vecs
            upper_levels[4] = level_base
            self.add(
                "alu",
                (
                    "+",
                    level4_addr,
                    self.scratch["forest_values_p"],
                    self.scratch_const(level4_start),
                ),
            )
            for v in range(2):
                self.add("load", ("vload", level4_values + v * VLEN, level4_addr))
                if v < 1:
                    self.add("flow", ("add_imm", level4_addr, level4_addr, VLEN))
            for p in range(level4_size):
                self.add(
                    "valu",
                    (
                        "vbroadcast",
                        level4_vecs + p * VLEN,
                        level4_values + p,
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
            level2_round=False,
            round_num=0,
        ):
            slots = []
            if (
                node_const is not None
                or node_pair is not None
                or node_arith is not None
                or node_prefetch is not None
            ):
                return slots

            if False and level2_round and level2_vecs_base is not None:

                slots.append(("alu", ("+", level2_addr_temp, self.scratch["forest_values_p"], level2_base_addr_const)))

                level2_scalars = [v_tmp1, v_tmp2, v_tmp3, v_tmp3]
                for i in range(4):
                    slots.append(("load", ("load", level2_scalars[i], level2_addr_temp)))
                    if i < 3:
                        slots.append(("flow", ("add_imm", level2_addr_temp, level2_addr_temp, 1)))

                level2_vecs = level2_vecs_base
                for i in range(4):
                    slots.append(("valu", ("vbroadcast", level2_vecs + i * VLEN, level2_scalars[i])))

                node_buf = v_node_block[buf_idx]
                for bi, vec_i in enumerate(block_vecs):
                    v_idx = idx_cache + vec_i
                    v_level_start = self.scratch_vconst(3)
                    v_offset = v_tmp2_block + bi * VLEN
                    v_bit0 = v_tmp1_block + bi * VLEN
                    v_bit1 = v_tmp3_block + bi * VLEN

                    slots.append(("valu", ("-", v_offset, v_idx, v_level_start)))
                    v_one = self.scratch_vconst(1)
                    slots.append(("valu", ("&", v_bit0, v_offset, v_one)))
                    slots.append(("valu", (">>", v_bit1, v_offset, self.scratch_vconst(1))))
                    slots.append(("valu", ("&", v_bit1, v_bit1, v_one)))

                    v_diff_01 = v_tmp1_block + bi * VLEN
                    v_diff_23 = v_tmp2_block + bi * VLEN
                    slots.append(("valu", ("-", v_diff_01, level2_vecs + 1 * VLEN, level2_vecs + 0 * VLEN)))
                    slots.append(("valu", ("-", v_diff_23, level2_vecs + 3 * VLEN, level2_vecs + 2 * VLEN)))

                    v_sel_low = v_tmp1_block + bi * VLEN
                    v_sel_high = v_tmp2_block + bi * VLEN
                    slots.append(("valu", ("multiply_add", v_sel_low, v_bit0, v_diff_01, level2_vecs + 0 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_sel_high, v_bit0, v_diff_23, level2_vecs + 2 * VLEN)))

                    v_result = v_tmp3_block + bi * VLEN
                    slots.append(("valu", ("-", v_result, v_sel_high, v_sel_low)))
                    slots.append(("valu", ("multiply_add", node_buf + bi * VLEN, v_bit1, v_result, v_sel_low)))

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
            slots = []
            for vec_i in block_vecs:
                v_idx = idx_cache + vec_i
                slots.append(("valu", ("+", v_addr[0], v_idx, v_forest_p)))
                for lane in range(VLEN):
                    slots.append(
                        ("load", ("load_offset", prefetch_base + vec_i, v_addr[0], lane))
                    )
            return slots

        def level2_prepare_slots():
            if not (enable_level2_where and enable_prefetch):
                return []
            slots = []
            slots.append(
                (
                    "alu",
                    (
                        "+",
                        level2_addr_temp,
                        self.scratch["forest_values_p"],
                        level2_base_addr_const,
                    ),
                )
            )
            for i in range(4):
                slots.append(("load", ("load", level2_scalars_base + i, level2_addr_temp)))
                if i < 3:
                    slots.append(("flow", ("add_imm", level2_addr_temp, level2_addr_temp, 1)))
            for i in range(4):
                slots.append(
                    ("valu", ("vbroadcast", level2_vecs_base + i * VLEN, level2_scalars_base + i))
                )
            return slots

        def level4_prepare_slots():
            if not (self.enable_level4_valu and enable_prefetch):
                return []
            slots = []
            level4_base_const = self.scratch_const(15)
            slots.append(
                (
                    "alu",
                    (
                        "+",
                        level2_addr_temp,
                        self.scratch["forest_values_p"],
                        level4_base_const,
                    ),
                )
            )
            for i in range(16):
                slots.append(("load", ("load", level2_scalars_base, level2_addr_temp)))
                slots.append(
                    ("valu", ("vbroadcast", level4_vecs_base + i * VLEN, level2_scalars_base))
                )
                if i < 15:
                    slots.append(("flow", ("add_imm", level2_addr_temp, level2_addr_temp, 1)))
            for pair in range(8):
                even = level4_vecs_base + (2 * pair) * VLEN
                odd = level4_vecs_base + (2 * pair + 1) * VLEN
                diff = level4_diffs_base + pair * VLEN
                slots.append(("valu", ("-", diff, odd, even)))
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
                elif level == 4:
                    v_level_start = self.scratch_vconst(15)
                    bases = level4_vecs_base
                    diffs = level4_diffs_base
                    v_t0 = t0
                    v_t1 = t1
                    v_t2 = t2
                    v_bit = bit
                    slots.append(("valu", ("-", v_t0, v_idx, v_level_start)))
                    slots.append(("valu", ("&", v_bit, v_t0, v_one)))
                    slots.append(("valu", ("multiply_add", v_t0, v_bit, diffs + 0 * VLEN, bases + 0 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_t1, v_bit, diffs + 1 * VLEN, bases + 2 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_t2, v_bit, diffs + 2 * VLEN, bases + 4 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_bit, v_bit, diffs + 3 * VLEN, bases + 6 * VLEN)))
                    slots.append(("valu", ("-", v_t1, v_t1, v_t0)))
                    slots.append(("valu", ("-", v_t2, v_t2, v_bit)))
                    slots.append(("valu", ("-", v_bit, v_idx, v_level_start)))
                    slots.append(("valu", (">>", v_bit, v_bit, v_one)))
                    slots.append(("valu", ("&", v_bit, v_bit, v_one)))
                    slots.append(("valu", ("multiply_add", v_t0, v_bit, v_t1, v_t0)))
                    slots.append(("valu", ("multiply_add", v_t2, v_bit, v_t2, v_bit)))
                    slots.append(("valu", ("-", v_t1, v_idx, v_level_start)))
                    slots.append(("valu", ("&", v_t1, v_t1, v_one)))
                    slots.append(("valu", ("multiply_add", v_t1, v_t1, diffs + 4 * VLEN, bases + 8 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_bit, v_t1, diffs + 5 * VLEN, bases + 10 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_t2, v_t1, diffs + 6 * VLEN, bases + 12 * VLEN)))
                    slots.append(("valu", ("multiply_add", v_t1, v_t1, diffs + 7 * VLEN, bases + 14 * VLEN)))
                    slots.append(("valu", ("-", v_bit, v_bit, v_t1)))
                    slots.append(("valu", ("-", v_t2, v_t2, v_t1)))
                    slots.append(("valu", ("-", v_t1, v_idx, v_level_start)))
                    slots.append(("valu", (">>", v_t1, v_t1, v_one)))
                    slots.append(("valu", ("&", v_t1, v_t1, v_one)))
                    slots.append(("valu", ("multiply_add", v_bit, v_t1, v_bit, v_t1)))
                    slots.append(("valu", ("multiply_add", v_t2, v_t1, v_t2, v_t1)))
                    slots.append(("valu", ("-", v_t1, v_idx, v_level_start)))
                    slots.append(("valu", (">>", v_t1, v_t1, v_two)))
                    slots.append(("valu", ("&", v_t1, v_t1, v_one)))
                    slots.append(("valu", ("-", v_t2, v_t2, v_t0)))
                    slots.append(("valu", ("multiply_add", v_t0, v_t1, v_t2, v_t0)))
                    slots.append(("valu", ("-", v_t2, v_bit, v_t0)))
                    slots.append(("valu", ("multiply_add", node, v_t1, v_t2, v_bit)))
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
            level2_round=False,
