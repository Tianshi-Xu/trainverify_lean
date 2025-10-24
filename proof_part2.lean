import trainverify.proof_def
import trainverify.proof_part1
import Mathlib.GroupTheory.Perm.Cycle.Basic
import Mathlib.GroupTheory.Perm.Cycle.Factors
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Finset.Basic

namespace TrainVerify

open scoped Classical

variable {input_dims output_dims : List ℕ}

/-- Cast a vector of natural numbers to integers elementwise. -/
def vectorToInt {n : ℕ} (v : List.Vector ℕ n) : List.Vector ℤ n :=
  List.Vector.map (fun x : ℕ => (x : ℤ)) v

@[simp]
lemma vectorToInt_get {n : ℕ} (v : List.Vector ℕ n) (t : Fin n) :
    (vectorToInt v).get t = (v.get t : ℤ) := by
  simp [vectorToInt]

/-- The elementwise `ℕ` → `ℤ` cast is injective on vectors. -/
@[simp]
lemma vectorToInt_eq_iff {n : ℕ} {v w : List.Vector ℕ n} :
    vectorToInt v = vectorToInt w ↔ v = w := by
  constructor
  · intro h
    apply List.Vector.ext
    intro t
    have := congrArg (fun vec => vec.get t) h
    simpa [vectorToInt_get] using this
  · intro h; simp [h]

/-- Natural-number sums equalities transfer to integer differences. -/
lemma sub_eq_sub_of_nat_add_eq {a b c d : ℕ} (h : a + b = c + d) :
    (a : ℤ) - (c : ℤ) = (d : ℤ) - (b : ℤ) := by
  apply (sub_eq_sub_iff_add_eq_add).2
  have h' : (a : ℤ) + (b : ℤ) = (c : ℤ) + (d : ℤ) := by
    have := congrArg (fun n : ℕ => (n : ℤ)) h
    simpa [Int.cast_add] using this
  simpa [add_comm] using h'.trans (by simp [add_comm, add_left_comm])

/-- Lean rendition of Lemma 5.5.3 (cycle decomposition) from `for_proof.md`.
Any finite permutation breaks into a list of pairwise-disjoint cycles whose product recovers the
original permutation.  We reuse Mathlib’s packaged construction `truncCycleFactors`. -/
lemma perm_cycle_decomposition {α : Type*} [Fintype α] [DecidableEq α]
    (σ : Equiv.Perm α) :
    ∃ L : List (Equiv.Perm α),
      L.prod = σ ∧ (∀ τ ∈ L, τ.IsCycle) ∧ L.Pairwise Equiv.Perm.Disjoint := by
  classical
  obtain ⟨L, hstruct⟩ := σ.truncCycleFactors.out
  rcases hstruct with ⟨hprod, hcycle, hpair⟩
  refine ⟨L, ?_⟩
  exact ⟨hprod, hcycle, hpair⟩
