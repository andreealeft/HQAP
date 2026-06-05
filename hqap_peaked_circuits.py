"""
A naive implementation of the HQAP (Heuristic Quantum Advantage with
Peaked Circuits) method from:

  Gharibyan et al., "Heuristic Quantum Advantage with Peaked Circuits"
  arXiv:2510.25838 (BlueQubit / Quantinuum, 2025)

Quantinuum H2 (56 qubits, ~2000 two-qubit gates).

----------------------------------
Remarks:
I used qiskit (statevector)
Don't try it on more than 6 qubits, or overall depth beyond 30 gates (the optimizier is too slow)
Currently working on locally perturbing U^dagger in various ways and testing against the marginal attack
----------------------------------

Section 3 of the paper
----------------------------------
  1. Train a shallow peaked circuit  R ▷ P(θ)
     R  = random circuit (the "scrambler")
     P  = variational peaking layer, trained so that one bitstring
          appears with anomalously high probability (peak weight δ_s)

  2. Insert an identity block  U ▷ U†  between R and P

  3. Obfuscate via Tensor Patch Optimisation (angle sweeping)
     Replace U† with a variational approximation ~U† that has high
     trace fidelity but hides the cancellation structure

  4. Apply swap transformations to further obscure qubit connectivity

  Final circuit:  swap(R) ▷ swap(U) ▷ ~U† ▷ P

  A verifier who knows the target bitstring can confirm the peak in
  O(1/δ_s) shots. A classical attacker must simulate the full circuit.

"""

import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from scipy.optimize import minimize

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to experiment
# ════════════════════════════════════════════════════════════════════════════

N_QUBITS      = 6           # number of qubits (paper: 56)
TARGET_BITS   = "010100"  # the secret peak bitstring
R_DEPTH       = 16          # RZZ gates in random layer R  (paper: ~2000)
P_DEPTH       = 3           # RZZ gates in peaking layer P
ID_DEPTH      = 6           # RZZ gates in identity block U
COBYLA_ITER   = 5000        # optimiser budget (stage 1)
NM_ITER       = 10000       # optimiser budget (stage 2)
SWEEP_ITER    = 5000        # sweep optimiser budget
N_SHOTS       = 1000        # simulated measurement shots
SEED          = 42

# ════════════════════════════════════════════════════════════════════════════
# HELPING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def bitstring_to_qiskit_index(bits: str) -> int:
    """
    Qiskit stores qubits in reversed bit order (qubit 0 = LSB).
    Convert a human-readable bitstring like "011001" to the matching
    statevector index.
    """
    return int(bits[::-1], 2)


def qiskit_index_to_bitstring(idx: int, n: int) -> str:
    return format(idx, f"0{n}b")[::-1]


def n_params(n: int, depth: int) -> int:
    """Parameters per circuit layer: 2 single-qubit angles per qubit + RZZ."""
    return depth * (2 * n + n // 2)


# ════════════════════════════════════════════════════════════════════════════
# CIRCUIT BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def random_rzz_circuit(n: int, n_rzz: int, rng: np.random.Generator) -> QuantumCircuit:
    """
    Build a random circuit with:
      - Rx, Rz single-qubit rotations on every qubit
      - One randomly-chosen RZZ entangling gate per layer

    This is the R circuit from the paper: a scrambling unitary that
    looks like a random circuit and is hard to classically simulate.
    """
    qc = QuantumCircuit(n)
    for _ in range(n_rzz):
        for q in range(n):
            qc.rx(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        pair = rng.choice(n, size=2, replace=False)
        qc.rzz(rng.uniform(0, np.pi), int(pair[0]), int(pair[1]))
    return qc


def variational_layer(n: int, depth: int, theta: np.ndarray) -> QuantumCircuit:
    """
    Build a variational circuit from parameter vector theta.
    Structure: Rx, Rz on each qubit, then RZZ on adjacent pairs.

    This is used both for the peaking layer P and for the approximate
    sweep ~U† in the tensor patch optimisation.

    Parameters
    ----------
    n     : number of qubits
    depth : number of RZZ layers
    theta : flat parameter vector of length n_params(n, depth)
    """
    qc = QuantumCircuit(n)
    idx = 0
    for _ in range(depth):
        for q in range(n):
            qc.rx(theta[idx], q); idx += 1
            qc.rz(theta[idx], q); idx += 1
        for q0, q1 in zip(range(0, n - 1, 2), range(1, n, 2)):
            qc.rzz(theta[idx], q0, q1); idx += 1
    return qc


def swap_obfuscate(qc: QuantumCircuit, swap_pairs: list) -> QuantumCircuit:
    """
    Section 3.2 — Swap Transformations.

    Wrap a circuit with SWAP gates:  SWAP ▷ circuit ▷ SWAP†
    This relabels qubit wires without changing the unitary action,
    but makes the connectivity pattern harder to analyse.
    """
    n = qc.num_qubits
    wrapper = QuantumCircuit(n)
    for a, b in swap_pairs:
        wrapper.swap(a, b)
    wrapper.compose(qc, inplace=True)
    for a, b in reversed(swap_pairs):
        wrapper.swap(a, b)
    return wrapper


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — TRAIN THE PEAKED CIRCUIT  R ▷ P(θ)
# ════════════════════════════════════════════════════════════════════════════

def train_peaked_circuit(
    R: QuantumCircuit,
    n: int,
    p_depth: int,
    target_idx: int,
    rng: np.random.Generator,
    cobyla_iter: int = 5000,
    nm_iter: int = 10000,
) -> tuple[np.ndarray, float]:
    """
    Optimise the peaking layer P so that R ▷ P peaks on target_idx.

    Loss = 1 - δ_s  where  δ_s = |<target|R▷P|0>|²  (peak weight).

    Two-stage optimisation (Section 3):
      Stage 1: COBYLA  — robust, derivative-free, good global search
      Stage 2: Nelder-Mead — refines to a tighter local minimum

    Returns
    -------
    theta_opt : optimal parameter vector
    peak_weight : achieved δ_s ∈ [0, 1]
    """
    def loss(theta: np.ndarray) -> float:
        P = variational_layer(n, p_depth, theta)
        sv = Statevector(R.compose(P))
        return 1.0 - float(sv.probabilities()[target_idx])

    theta0 = rng.uniform(0, 2 * np.pi, n_params(n, p_depth))

    print("  Stage 1: COBYLA optimisation ...")
    r1 = minimize(loss, theta0, method="COBYLA",
                  options={"maxiter": cobyla_iter, "rhobeg": 0.5})

    print("  Stage 2: Nelder-Mead refinement ...")
    r2 = minimize(loss, r1.x, method="Nelder-Mead",
                  options={"maxiter": nm_iter, "xatol": 1e-6, "fatol": 1e-6})

    return r2.x, 1.0 - r2.fun


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD IDENTITY BLOCK  U ▷ U†
# ════════════════════════════════════════════════════════════════════════════

def build_identity_block(n: int, depth: int, rng: np.random.Generator):
    """
    Build U and U† (exact inverse).  Inserted between R and P so that
    the circuit's peak is preserved:

        R ▷ [U ▷ U†] ▷ P  =  R ▷ P   (exactly)

    The obfuscation in Step 3 breaks this exact cancellation slightly.
    """
    U = random_rzz_circuit(n, depth, rng)
    Udagger = U.inverse()
    return U, Udagger


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — TENSOR PATCH OPTIMISATION (angle sweeping)
# ════════════════════════════════════════════════════════════════════════════

def sweep_unitary(
    target_circuit: QuantumCircuit,
    n: int,
    sweep_depth: int,
    rng: np.random.Generator,
    max_iter: int = 5000,
) -> tuple[QuantumCircuit, float]:
    """
    Section 3.1 — Tensor Patch Optimisation (angle sweeping).

    Train a new variational circuit ~V to approximate target_circuit
    by minimising the trace-fidelity loss:

        L(θ) = 1 - |Tr[U†·~V(θ)]| / d      (Eq. 1 in paper)

    The result looks like V but has different gate angles, hiding the
    relationship between U and U† in the identity block.

    Parameters
    ----------
    target_circuit : the circuit to approximate (~U†)
    sweep_depth    : number of layers in the approximation
    """
    target_mat = Operator(target_circuit).data
    d = 2 ** n

    def patch_loss(theta: np.ndarray) -> float:
        approx = variational_layer(n, sweep_depth, theta)
        op = Operator(approx).data
        return 1.0 - abs(np.trace(target_mat.conj().T @ op)) / d

    # better would be: start from optimal angles (that we know from U)
    # and then perturb them a bit; start from a wanted precision and determine the angles?
    theta0 = rng.uniform(0, 2 * np.pi, n_params(n, sweep_depth))
    result = minimize(patch_loss, theta0, method="COBYLA",
                      options={"maxiter": max_iter, "rhobeg": 0.5})
    result2 = minimize(patch_loss, result.x, method="Nelder-Mead",
                       options={"maxiter": max_iter})

    swept_circuit = variational_layer(n, sweep_depth, result2.x)
    fidelity = 1.0 - result2.fun
    return swept_circuit, fidelity


# ════════════════════════════════════════════════════════════════════════════
# VERIFICATION — simulate "quantum device" via statevector sampling
# ════════════════════════════════════════════════════════════════════════════

def simulate_quantum_device(
    circuit: QuantumCircuit,
    target_idx: int,
    n: int,
    n_shots: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate running the circuit on a quantum device:
      1. Compute the exact output probability distribution (statevector)
      2. Sample n_shots measurements from it
      3. Report the most frequent bitstring
    """
    sv = Statevector(circuit)
    probs = sv.probabilities()

    # Sample measurements
    counts_arr = rng.multinomial(n_shots, probs)
    top_idx = int(np.argmax(counts_arr))

    top5 = np.argsort(probs)[::-1][:5]

    return {
        "peak_weight":        float(probs[target_idx]),
        "top_bitstring":      qiskit_index_to_bitstring(top_idx, n),
        "top_bitstring_count": int(counts_arr[top_idx]),
        "target_count":       int(counts_arr[target_idx]),
        "top5": [
            (qiskit_index_to_bitstring(i, n), float(probs[i]))
            for i in top5
        ],
    }

# ════════════════════════════════════════════════════════════════════════════
# CLASSICAL ATTACK — marginal attack (Section 4.1)
# ════════════════════════════════════════════════════════════════════════════

def marginal_attack(circuit: QuantumCircuit, n: int) -> str:
    """
    Section 4.1 — Marginal Attack Strategy.

    The classical attacker computes the single-qubit expectation value
    ⟨Z_i⟩ for each qubit and reads off the bit from its sign:

        p(b_i = 0) = (1 + ⟨Z_i⟩) / 2     (Eq. 3)
        b_i = 0  if  ⟨Z_i⟩ > 0,  else 1

    For small circuits this is tractable (full statevector).
    For the paper's 56-qubit / 2000-gate circuits, even computing ⟨Z_i⟩
    approximately (via MPS/TN/PPS) becomes infeasible beyond ~700 gates.

    I use the exact statevector for now (because I don't know yet how to do it fancier).
    """
    sv = Statevector(circuit)
    probs = sv.probabilities()

    guessed_bits = []
    for q in range(n):
        # Marginal probability that qubit q = 0
        # Sum probabilities of all basis states where bit q = 0
        p0 = sum(
            probs[idx]
            for idx in range(2 ** n)
            if not (idx >> q & 1)      # bit q is 0 in this basis state
        )
        z_exp = 2 * p0 - 1            # ⟨Z_q⟩ = p(0) - p(1)
        guessed_bits.append("0" if z_exp > 0 else "1")

    # Qiskit bit ordering: qubit 0 is index 0, so no reversal needed here
    return "".join(guessed_bits)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("HQAP Peaked Circuit")
    print("Based on arXiv:2510.25838 (BlueQubit / Quantinuum, 2025)")
    print("=" * 60)
    print(f"\nConfig: {N_QUBITS} qubits, target='{TARGET_BITS}', "
          f"R_depth={R_DEPTH}, P_depth={P_DEPTH}, ID_depth={ID_DEPTH}")

    assert len(TARGET_BITS) == N_QUBITS, \
        f"TARGET_BITS length {len(TARGET_BITS)} must equal N_QUBITS {N_QUBITS}"

    target_idx = bitstring_to_qiskit_index(TARGET_BITS)

    # ── Step 1: Train R ▷ P ─────────────────────────────────────────────
    t0 = time.time()
    print(f"\n[Step 1] Training peaked circuit R ▷ P  (target: '{TARGET_BITS}')")
    R = random_rzz_circuit(N_QUBITS, R_DEPTH, rng)
    theta_opt, peak_weight_trained = train_peaked_circuit(
        R, N_QUBITS, P_DEPTH, target_idx, rng,
        cobyla_iter=COBYLA_ITER, nm_iter=NM_ITER,
    )
    P_final = variational_layer(N_QUBITS, P_DEPTH, theta_opt)
    clean_circuit = R.compose(P_final)
    print(f"  ✓ Achieved peak weight δ_s = {peak_weight_trained:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # Verify clean circuit
    print(f"\n[Verification] Clean circuit R ▷ P  (pre-obfuscation)")
    clean_result = simulate_quantum_device(
        clean_circuit, target_idx, N_QUBITS, N_SHOTS, rng
    )
    print(f"  Peak weight:   {clean_result['peak_weight']:.4f}")
    print(f"  Most sampled:  '{clean_result['top_bitstring']}'  "
          f"({clean_result['top_bitstring_count']}/{N_SHOTS} shots)")
    print(f"  Target hits:   {clean_result['target_count']}/{N_SHOTS} shots")
    print(f"  Top 5 bitstrings:")
    for bs, p in clean_result["top5"]:
        marker = " ← TARGET" if bs == TARGET_BITS else ""
        print(f"    {bs}  {p:.4f}{marker}")

    # ── Step 2: Identity block ───────────────────────────────────────────
    print(f"\n[Step 2] Building identity block U ▷ U†  (depth={ID_DEPTH})")
    U, Udagger = build_identity_block(N_QUBITS, ID_DEPTH, rng)

    # Sanity check: R ▷ U ▷ U† ▷ P should be identical to R ▷ P
    exact_circuit = R.compose(U).compose(Udagger).compose(P_final)
    exact_result = simulate_quantum_device(
        exact_circuit, target_idx, N_QUBITS, N_SHOTS, rng
    )
    print(f"  ✓ Identity block preserves peak weight: "
          f"{exact_result['peak_weight']:.4f}  (should match {peak_weight_trained:.4f})")

    # ── Step 3: Tensor Patch Optimisation (angle sweeping) ───────────────
    print(f"\n[Step 3] Tensor Patch Optimisation — sweep U†  (depth={ID_DEPTH})")
    swept_Udagger, sweep_fidelity = sweep_unitary(
        Udagger, N_QUBITS, ID_DEPTH, rng, max_iter=SWEEP_ITER
    )
    print(f"  Sweep fidelity |Tr[U†·~U†]|/d = {sweep_fidelity:.4f}")
    print(f"  (1.0 = exact copy; lower = more obfuscated but peak degrades)")

    # ── Step 4: Swap transformations ─────────────────────────────────────
    print(f"\n[Step 4] Swap obfuscation")
    swap_pairs = [(0, N_QUBITS - 1), (1, N_QUBITS - 2)]
    R_obf  = swap_obfuscate(R,  swap_pairs)
    U_obf  = swap_obfuscate(U,  swap_pairs)
    print(f"  Applied {len(swap_pairs)} swap pair(s): {swap_pairs}")

    # ── Assemble final circuit  T[R] ▷ T[U] ▷ ~U† ▷ P ──────────────────
    print(f"\n[Assembly] Final circuit: swap(R) ▷ swap(U) ▷ ~U† ▷ P")
    obf_circuit = R_obf.compose(U_obf).compose(swept_Udagger).compose(P_final)
    gate_counts = dict(obf_circuit.count_ops())
    print(f"  Gate counts: {gate_counts}")
    print(f"  Total 2-qubit gates: "
          f"{sum(v for k,v in gate_counts.items() if k in ('rzz','cx','swap'))}")

    # ── Verification on obfuscated circuit ───────────────────────────────
    print(f"\n[Verification] Obfuscated circuit  (quantum device simulation)")
    obf_result = simulate_quantum_device(
        obf_circuit, target_idx, N_QUBITS, N_SHOTS, rng
    )
    print(f"  Peak weight at target: {obf_result['peak_weight']:.4f}")
    print(f"  Target hits:  {obf_result['target_count']}/{N_SHOTS} shots")
    print(f"  Top 5 bitstrings:")
    for bs, p in obf_result["top5"]:
        marker = " ← TARGET" if bs == TARGET_BITS else ""
        print(f"    {bs}  {p:.4f}{marker}")

    # ── Classical marginal attack ─────────────────────────────────────────
    print(f"\n[Classical Attack] Marginal attack on obfuscated circuit (Section 4.1)")
    print(f"  Computing ⟨Z_i⟩ for each qubit ...")
    guessed = marginal_attack(obf_circuit, N_QUBITS)
    correct_bits = sum(g == t for g, t in zip(guessed, TARGET_BITS))
    print(f"  Target:  {TARGET_BITS}")
    print(f"  Guessed: {guessed}")
    print(f"  Correct bits: {correct_bits}/{N_QUBITS}  "
          f"({'✓ CRACKED' if guessed == TARGET_BITS else '✗ failed'})")
    print(f"\n  NOTE: At 6 qubits, the marginal attack uses exact statevector")
    print(f"  simulation (feasible).  At 56 qubits with 2000 gates the same")
    print(f"  attack requires MPS/TN/PPS and fails beyond ~700 gates.")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Target bitstring:              '{TARGET_BITS}'")
    print(f"  Peak weight (clean circuit):   {peak_weight_trained:.4f}  "
          f"({'strong' if peak_weight_trained > 0.5 else 'moderate'} peak)")
    print(f"  Peak weight (obfuscated):      {obf_result['peak_weight']:.4f}  "
          f"(sweep fidelity={sweep_fidelity:.3f})")
    print(f"  Quantum verification:          "
          f"{'PASS' if clean_result['top_bitstring'] == TARGET_BITS else 'FAIL'} "
          f"(clean circuit)")
    print(f"  Classical marginal attack:     "
          f"{'PASS' if guessed == TARGET_BITS else 'FAIL (obfuscation effective at scale)'}")
    print()
    print("  Paper scale vs this demo:")
    print(f"    N qubits:       {N_QUBITS}  (paper: 56)")
    print(f"    2-qubit gates:  {R_DEPTH + P_DEPTH}  (paper: 2000)")
    print(f"    Classical sim:  exact statevector  (paper: MPS/TN+BP/PPS)")
    print(f"    Quantum device: Qiskit simulator  (paper: Quantinuum H2)")
    total_time = time.time() - t0
    print(f"\n  Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
