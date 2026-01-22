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

from collections import defaultdict
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
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

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

        def flush():
            nonlocal bundle, bundle_reads, bundle_writes
            if bundle:
                instrs.append(bundle)
            bundle = {}
            bundle_reads = set()
            bundle_writes = set()

        for engine, slot in slots:
            reads, writes, barrier = self._slot_reads_writes(engine, slot)
            if barrier:
                flush()
                instrs.append({engine: [slot]})
                continue

            if len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]:
                flush()
            if reads & bundle_writes:
                flush()
            if writes & bundle_writes:
                flush()

            bundle.setdefault(engine, []).append(slot)
            bundle_reads.update(reads)
            bundle_writes.update(writes)

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
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2):
        slots = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            slots.append(("valu", (op1, tmp1, val_hash_addr, self.scratch_vconst(val1))))
            slots.append(("valu", (op3, tmp2, val_hash_addr, self.scratch_vconst(val3))))
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

        # Precompute input pointers to avoid repeated address arithmetic in the hot loop.
        idx_ptrs = self.alloc_scratch("idx_ptrs", batch_size)
        val_ptrs = self.alloc_scratch("val_ptrs", batch_size)
        self.add("alu", ("+", idx_ptrs, self.scratch["inp_indices_p"], zero_const))
        self.add("alu", ("+", val_ptrs, self.scratch["inp_values_p"], zero_const))
        for i in range(1, batch_size):
            self.add("alu", ("+", idx_ptrs + i, idx_ptrs + i - 1, one_const))
            self.add("alu", ("+", val_ptrs + i, val_ptrs + i - 1, one_const))

        # Vector scratch registers and broadcasted constants.
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)
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

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            vec_end = batch_size - (batch_size % VLEN)
            for i in range(0, vec_end, VLEN):
                idx_ptr = idx_ptrs + i
                val_ptr = val_ptrs + i
                body.append(("load", ("vload", v_idx, idx_ptr)))
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_idx,
                            [(round, i + lane, "idx") for lane in range(VLEN)],
                        ),
                    )
                )
                body.append(("load", ("vload", v_val, val_ptr)))
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_val,
                            [(round, i + lane, "val") for lane in range(VLEN)],
                        ),
                    )
                )
                body.append(("valu", ("+", v_addr, v_idx, v_forest_p)))
                for lane in range(VLEN):
                    body.append(("load", ("load_offset", v_node_val, v_addr, lane)))
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_node_val,
                            [(round, i + lane, "node_val") for lane in range(VLEN)],
                        ),
                    )
                )
                body.append(("valu", ("^", v_val, v_val, v_node_val)))
                body.extend(self.build_hash_vec(v_val, v_tmp1, v_tmp2))
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_val,
                            [(round, i + lane, "hashed_val") for lane in range(VLEN)],
                        ),
                    )
                )
                body.append(("valu", ("&", v_tmp3, v_val, v_one)))
                body.append(("valu", ("+", v_tmp3, v_tmp3, v_one)))
                body.append(("valu", ("+", v_idx, v_idx, v_idx)))
                body.append(("valu", ("+", v_idx, v_idx, v_tmp3)))
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_idx,
                            [(round, i + lane, "next_idx") for lane in range(VLEN)],
                        ),
                    )
                )
                body.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
                body.append(("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero)))
                body.append(
                    (
                        "debug",
                        (
                            "vcompare",
                            v_idx,
                            [(round, i + lane, "wrapped_idx") for lane in range(VLEN)],
                        ),
                    )
                )
                body.append(("store", ("vstore", idx_ptr, v_idx)))
                body.append(("store", ("vstore", val_ptr, v_val)))

            for i in range(vec_end, batch_size):
                idx_ptr = idx_ptrs + i
                val_ptr = val_ptrs + i
                # idx = mem[inp_indices_p + i]
                body.append(("load", ("load", tmp_idx, idx_ptr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("load", ("load", tmp_val, val_ptr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                )
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(
                    ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                )
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("&", tmp3, tmp_val, one_const)))
                body.append(("alu", ("+", tmp3, tmp3, one_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("store", ("store", idx_ptr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("store", ("store", val_ptr, tmp_val)))

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
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
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
