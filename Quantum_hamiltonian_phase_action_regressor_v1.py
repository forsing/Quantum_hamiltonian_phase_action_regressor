# Quantum_hamiltonian_phase_action_regressor



import random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
TAU_GRID = (0.35, 0.70)
REG_GRID = (1e-3, 1e-2, 5e-2)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < 35:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_phase_space_transitions(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    q_list = []
    p_list = []
    y_list = []
    for t in range(2, n_total):
        x_prev2 = draws.__getitem__(t - 2).astype(np.float64) / float(MAX_NUM)
        x_prev1 = draws.__getitem__(t - 1).astype(np.float64) / float(MAX_NUM)
        q_t = x_prev1
        p_t = x_prev1 - x_prev2
        q_list.append(q_t)
        p_list.append(p_t)
        y_list.append(draws.__getitem__(t))
    return (
        np.array(q_list, dtype=np.float64),
        np.array(p_list, dtype=np.float64),
        np.array(y_list, dtype=np.int64),
    )


def pauli_string_two_qubits(q_a: int, q_b: int, op_a: str, op_b: str) -> str:
    chars = ["I"] * NUM_QUBITS
    chars[NUM_QUBITS - 1 - q_a] = op_a
    chars[NUM_QUBITS - 1 - q_b] = op_b
    return "".join(chars)


def pauli_string_one_qubit(q_idx: int, op: str) -> str:
    chars = ["I"] * NUM_QUBITS
    chars[NUM_QUBITS - 1 - q_idx] = op
    return "".join(chars)


def build_hamiltonian_operator(q_vec: np.ndarray, p_vec: np.ndarray) -> SparsePauliOp:
    terms: list[tuple[str, float]] = []

    for q in range(NUM_QUBITS):
        coord_val = float(q_vec.__getitem__(q)) - 0.5
        terms.append((pauli_string_one_qubit(q, "Z"), coord_val))

    for q in range(NUM_QUBITS - 1):
        pair_pot = float(q_vec.__getitem__(q)) * float(q_vec.__getitem__(q + 1)) - 0.25
        terms.append((pauli_string_two_qubits(q, q + 1, "Z", "Z"), 0.5 * pair_pot))

        kin_mom = float(p_vec.__getitem__(q)) - float(p_vec.__getitem__(q + 1))
        terms.append((pauli_string_two_qubits(q, q + 1, "X", "X"), 0.5 * kin_mom))
        terms.append((pauli_string_two_qubits(q, q + 1, "Y", "Y"), 0.5 * kin_mom))

    ring_pot = float(q_vec.__getitem__(NUM_QUBITS - 1)) * float(q_vec.__getitem__(NUM_POSITIONS - 1)) - 0.25
    terms.append((pauli_string_two_qubits(NUM_QUBITS - 1, 0, "Z", "Z"), 0.5 * ring_pot))

    ring_mom = float(p_vec.__getitem__(NUM_POSITIONS - 1))
    terms.append((pauli_string_two_qubits(NUM_QUBITS - 1, 0, "X", "X"), 0.5 * ring_mom))
    terms.append((pauli_string_two_qubits(NUM_QUBITS - 1, 0, "Y", "Y"), 0.5 * ring_mom))

    return SparsePauliOp.from_list(terms).simplify()


def build_trotter_hamiltonian_circuit(
    q_row: np.ndarray,
    p_row: np.ndarray,
    tau: float,
    reps: int = 2,
) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        qc.ry(np.pi * (float(q_row.__getitem__(q)) - 0.5), q)
        qc.rz(0.5 * np.pi * float(p_row.__getitem__(q)), q)

    dt = tau / float(reps)
    for _ in range(reps):
        for q in range(NUM_QUBITS):
            coord_val = float(q_row.__getitem__(q)) - 0.5
            qc.rz(2.0 * dt * coord_val, q)

        for q in range(NUM_QUBITS - 1):
            pair_pot = float(q_row.__getitem__(q)) * float(q_row.__getitem__(q + 1)) - 0.25
            qc.rzz(dt * pair_pot, q, q + 1)

            kin_mom = float(p_row.__getitem__(q)) - float(p_row.__getitem__(q + 1))
            qc.rxx(dt * kin_mom, q, q + 1)
            qc.ryy(dt * kin_mom, q, q + 1)

        ring_pot = float(q_row.__getitem__(NUM_QUBITS - 1)) * float(q_row.__getitem__(NUM_POSITIONS - 1)) - 0.25
        qc.rzz(dt * ring_pot, NUM_QUBITS - 1, 0)

        ring_mom = float(p_row.__getitem__(NUM_POSITIONS - 1))
        qc.rxx(dt * ring_mom, NUM_QUBITS - 1, 0)
        qc.ryy(dt * ring_mom, NUM_QUBITS - 1, 0)

    return qc


def evolve_phase_space_amplitudes(
    q_mat: np.ndarray,
    p_mat: np.ndarray,
    tau: float,
) -> np.ndarray:
    n_samples = int(q_mat.shape.__getitem__(0))
    init_qc = QuantumCircuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        init_qc.h(q)
    base_sv = Statevector.from_int(0, dims=STATE_DIM).evolve(init_qc)

    _ = build_hamiltonian_operator(q_mat.__getitem__(0), p_mat.__getitem__(0))
    amp_feats = np.empty((n_samples, 3 * STATE_DIM), dtype=np.float64)

    for i in range(n_samples):
        q_row = q_mat.__getitem__(i)
        p_row = p_mat.__getitem__(i)
        prep_qc = build_trotter_hamiltonian_circuit(q_row, p_row, tau=tau, reps=2)

        evolved_sv = base_sv.evolve(prep_qc).data
        re = np.real(evolved_sv)
        im = np.imag(evolved_sv)
        prob = np.abs(evolved_sv) ** 2
        amp_feats[i] = np.concatenate([re, im, prob])

    return amp_feats


def get_position_mask() -> np.ndarray:
    num_grid = np.arange(1, MAX_NUM + 1, dtype=np.int64)[None, :]
    pos_idx = np.arange(NUM_POSITIONS, dtype=np.int64)[:, None]
    min_allowed = pos_idx + 1
    max_allowed = MAX_NUM - (NUM_POSITIONS - 1 - pos_idx)
    return (num_grid >= min_allowed) & (num_grid <= max_allowed)


def amplitudes_to_energy_gibbs_probs(
    amp_feats: np.ndarray,
    w_energy: np.ndarray,
    b_energy: np.ndarray,
    pos_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    energies = np.einsum("bf,pfk->bpk", amp_feats, w_energy) + b_energy[None, :, :]
    neg_e = -energies
    neg_e_masked = np.where(pos_mask[None, :, :], neg_e, -1e9)
    neg_e_shifted = neg_e_masked - np.max(neg_e_masked, axis=2, keepdims=True)
    boltzmann = np.where(pos_mask[None, :, :], np.exp(np.clip(neg_e_shifted, -30.0, 30.0)), 0.0)
    z_part = np.sum(boltzmann, axis=2, keepdims=True)
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]
    probs = np.where(z_part > 1e-15, boltzmann / np.maximum(z_part, 1e-15), uniform_fb)
    return probs, energies


def fit_hamiltonian_potential_regressor(
    amp_train: np.ndarray,
    y_train: np.ndarray,
    reg_weight: float,
    maxiter: int = 55,
) -> dict:
    rng = np.random.default_rng(39)
    n_feats = int(amp_train.shape.__getitem__(1))
    w_shape = (NUM_POSITIONS, n_feats, MAX_NUM)
    b_shape = (NUM_POSITIONS, MAX_NUM)

    n_w = int(np.prod(w_shape))
    n_b = int(np.prod(b_shape))

    init_w = rng.normal(0.0, 0.03, size=n_w)
    init_b = np.zeros(n_b, dtype=np.float64)
    init_params = np.concatenate([init_w, init_b])

    pos_mask = get_position_mask()
    n_samples = len(y_train)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_train - 1

    one_hot_y = np.zeros((n_samples, NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    one_hot_y[sample_idx, pos_idx, target_idx] = 1.0
    n_total_params = float(len(init_params))

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        w_e = params[:n_w].reshape(w_shape)
        b_e = params[n_w:].reshape(b_shape)

        probs, _ = amplitudes_to_energy_gibbs_probs(amp_train, w_e, b_e, pos_mask)
        chosen = probs[sample_idx, pos_idx, target_idx]
        nll = -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))
        reg = reg_weight * float(np.mean(params ** 2))
        total_loss = nll + reg

        diff_energy = (one_hot_y - probs) * pos_mask[None, :, :] / float(n_samples)
        grad_w = np.einsum("bf,bpk->pfk", amp_train, diff_energy) + (2.0 * reg_weight / n_total_params) * w_e
        grad_b = np.sum(diff_energy, axis=0) + (2.0 * reg_weight / n_total_params) * b_e

        return total_loss, np.concatenate([grad_w.reshape(-1), grad_b.reshape(-1)])

    res = minimize(
        loss_and_grad,
        init_params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-4},
    )

    opt = res.x
    return {
        "w_energy": opt[:n_w].reshape(w_shape),
        "b_energy": opt[n_w:].reshape(b_shape),
        "pos_mask": pos_mask,
    }


def evaluate_gibbs_nll(model: dict, amp_eval: np.ndarray, y_eval: np.ndarray) -> float:
    probs, _ = amplitudes_to_energy_gibbs_probs(
        amp_eval,
        model.__getitem__("w_energy"),
        model.__getitem__("b_energy"),
        model.__getitem__("pos_mask"),
    )
    n_samples = len(y_eval)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_eval - 1
    chosen = probs[sample_idx, pos_idx, target_idx]
    return -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))









def least_action_dp_decode(energies_single: np.ndarray, pos_mask: np.ndarray) -> list[int]:
    inf_val = 1e18
    dp = np.full((NUM_POSITIONS, MAX_NUM), inf_val, dtype=np.float64)
    parent = np.full((NUM_POSITIONS, MAX_NUM), -1, dtype=np.int64)

    first_mask = pos_mask.__getitem__(0)
    first_e = energies_single.__getitem__(0)
    for k in range(MAX_NUM):
        if bool(first_mask.__getitem__(k)):
            dp[0, k] = float(first_e.__getitem__(k))

    for pos in range(1, NUM_POSITIONS):
        curr_mask = pos_mask.__getitem__(pos)
        curr_e = energies_single.__getitem__(pos)
        prev_dp = dp.__getitem__(pos - 1)
        for k in range(MAX_NUM):
            if not bool(curr_mask.__getitem__(k)):
                continue
            best_prev_cost = inf_val
            best_prev_k = -1
            for prev_k in range(k):
                prev_cost = float(prev_dp.__getitem__(prev_k))
                if prev_cost < best_prev_cost:
                    best_prev_cost = prev_cost
                    best_prev_k = prev_k
            if best_prev_k >= 0 and best_prev_cost < inf_val * 0.5:
                dp[pos, k] = best_prev_cost + float(curr_e.__getitem__(k))
                parent[pos, k] = best_prev_k

    last_dp = dp.__getitem__(NUM_POSITIONS - 1)
    best_last_k = int(np.argmin(last_dp))
    if float(last_dp.__getitem__(best_last_k)) >= inf_val * 0.5:
        raise RuntimeError("Neuspesno dekodiranje putanje najmanjeg dejstva.")

    seq_rev: list[int] = []
    curr_k = best_last_k
    for pos in range(NUM_POSITIONS - 1, -1, -1):
        seq_rev.append(curr_k + 1)
        curr_k = int(parent[pos, curr_k])

    seq = list(reversed(seq_rev))
    if len(set(seq)) != NUM_POSITIONS:
        raise RuntimeError("Dekodirana putanja nema 7 razlicitih brojeva.")
    return seq


def main() -> None:
    print("Metoda: Qiskit Hamiltonian Phase-Space & Least-Action Regressor (1x6 qubita)")
    print("Posebnost: Trotter-Suzuki evolucija H(q,p)=T(p)+V(q) na simulatoru + DP dekodiranje najmanjeg dejstva")

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        q_all, p_all, y_all = build_phase_space_transitions(draws)
        n_samples = int(y_all.shape.__getitem__(0))

        train_end = int(0.75 * n_samples)
        val_end = int(0.88 * n_samples)

        y_tr = y_all[:train_end]
        y_va = y_all[train_end:val_end]
        y_te = y_all[val_end:]

        best_tau = float(TAU_GRID.__getitem__(0))
        best_reg = float(REG_GRID.__getitem__(0))
        best_val_nll = float("inf")
        cached_amps: dict[float, np.ndarray] = {}

        for tau in TAU_GRID:
            amp_all_tau = evolve_phase_space_amplitudes(q_all, p_all, tau=tau)
            cached_amps[tau] = amp_all_tau
            amp_tr = amp_all_tau[:train_end]
            amp_va = amp_all_tau[train_end:val_end]

            for reg in REG_GRID:
                model = fit_hamiltonian_potential_regressor(amp_tr, y_tr, reg_weight=reg, maxiter=55)
                val_nll = evaluate_gibbs_nll(model, amp_va, y_va)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_tau = tau
                    best_reg = reg

        amp_best = cached_amps.__getitem__(best_tau)
        eval_model = fit_hamiltonian_potential_regressor(
            amp_best[:train_end],
            y_tr,
            reg_weight=best_reg,
            maxiter=55,
        )
        test_nll = evaluate_gibbs_nll(eval_model, amp_best[val_end:], y_te)

        final_model = fit_hamiltonian_potential_regressor(
            amp_best,
            y_all,
            reg_weight=best_reg,
            maxiter=55,
        )

        x_prev2 = draws.__getitem__(len(draws) - 2).astype(np.float64) / float(MAX_NUM)
        x_prev1 = draws.__getitem__(len(draws) - 1).astype(np.float64) / float(MAX_NUM)
        q_next = x_prev1[None, :]
        p_next = (x_prev1 - x_prev2)[None, :]
        amp_next = evolve_phase_space_amplitudes(q_next, p_next, tau=best_tau)

        _, energies_next = amplitudes_to_energy_gibbs_probs(
            amp_next,
            final_model.__getitem__("w_energy"),
            final_model.__getitem__("b_energy"),
            final_model.__getitem__("pos_mask"),
        )
        next_combo = least_action_dp_decode(
            energies_next.__getitem__(0),
            final_model.__getitem__("pos_mask"),
        )

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (tau={best_tau}, reg={best_reg}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Qiskit Hamiltonian Phase-Space & Least-Action Regressor (1x6 qubita)
Posebnost: Trotter-Suzuki evolucija H(q,p)=T(p)+V(q) na simulatoru + DP dekodiranje najmanjeg dejstva

CSV: loto7_4718_k90.csv
Validacija NLL (tau=0.35, reg=0.01): 21.0162
Test NLL: 20.9688
NEXT: [1, x, 18, y, 23, z, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (tau=0.35, reg=0.05): 21.3679
Test NLL: 21.5938
NEXT: [4, x, 10, y, 26, z, 38]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (tau=0.35, reg=0.05): 21.3088
Test NLL: 21.7915
NEXT: [7, x, 12, y, 29, z, 37]
"""



"""
Lagrangian mechanics and Hamiltonian phase space

	• U Lagranžovoj mehanici, stanje dinamičkog sistema opisuje se generalisanim koordinatama \(q_{t}\) i generalisanim brzinama \(\dot{q}_t \approx q_t - q_{t-1}\), pri čemu se stvarna putanja sistema kreće tako da minimizuje dejstvo (Action) \(S = \int L(q, \dot{q})\,dt = \int (T(\dot{q}) - V(q))\,dt\) (kinetička minus potencijalna energija).
	• Prelaskom u Hamiltonov fazni prostor \((q_t, p_t)\), gde je kanonski impuls \(p_t = \partial L / \partial \dot{q}_t\), evolucija sistema očuvava simplektičku strukturu i totalnu energiju \(H(q, p) = T(p) + V(q)\) putem Hamiltonovih jednačina \(\dot{q} = \partial H / \partial p\), \(\dot{p} = -\partial H / \partial q\) (i njihove kvantne analogije \(U(\Delta t) = e^{-i \hat{H} \Delta t}\)).
    • Hamiltonova dinamika ima prirodnu, direktnu vezu sa kvantnom mehanikom kroz unitarnu evoluciju stanja pod dejstvom kvantnog Hamiltonijana \(\^{H}\) (SparsePauliOp + PauliEvolutionGate / Trotter-Suzuki evolucija na 1x6 qubita, rad sa amplitudama na lokalnom simulatoru):
	
    1. Koordinate i kanonski impuls 
	2. Kvantni Hamiltonijan 
	3. Evolucija amplituda i princip najmanjeg dejstva 
    
Najprirodniji most između klasične analitičke mehanike i kvantne evolucije amplituda u Qiskit-u.
Hamiltonov formalizam eksplicitno spaja položaj (\(q_{t}\)) i brzinu promene / impuls (\(p_{t}\)) u zajednički fazni prostor i evoluira ih kroz kinetičko-potencijalni kvantni Hamiltonijan \(\hat{H} = \hat{T} + \hat{V}\).
Dekodiranje putem principa najmanjeg dejstva (minimizacija ukupne energije prelaza od 1. do 7. broja uz strogi rastući poredak) daje potpuno nov, fizički utemeljen mehanizam izbora NEXT kombinacije.
"""
