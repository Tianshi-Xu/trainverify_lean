import Mathlib.Data.Vector.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Range
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Data.Vector.Basic -- For Vector.get_eq_get_toList

namespace SIMD

-- Redefine basic types to avoid import issues
/-- A tensor is represented as a function from multi-dimensional indices to real numbers -/
def Tensor (dims : List ℕ) : Type :=
  (List.Vector ℕ dims.length) → ℝ

/-- Multi-dimensional index type -/
def Index (n : ℕ) : Type := List.Vector ℕ n

/-- Check if an index is valid for given dimensions -/
def validIndex (dims : List ℕ) (idx : Index dims.length) : Prop :=
  ∀ i : Fin dims.length, idx.get i < dims.get i

/-- Kernel function: takes k scalar inputs and produces one scalar output -/
def KernelFunction (k : ℕ) : Type :=
  (List.Vector ℝ k) → ℝ

/-- Multi-tensor input structure: a tuple of p tensors with their respective dimensions -/
structure MultiTensorInput where
  p : ℕ  -- Number of input tensors
  dims : List.Vector (List ℕ) p  -- Dimensions of each input tensor
  tensors : (i : Fin p) → Tensor (dims.get i)  -- The actual tensors

/-- Input pointer: (tensor_idx, multi_dim_idx) to uniquely locate an element -/
structure InputPointer (input : MultiTensorInput) where
  tensor_idx : Fin input.p  -- Which tensor (0 ≤ tensor_idx < p)
  multi_dim_idx : Index (input.dims.get tensor_idx).length  -- Multi-dimensional index within that tensor

/-- Check if an input pointer is valid -/
def validInputPointer (input : MultiTensorInput) (ptr : InputPointer input) : Prop :=
  validIndex (input.dims.get ptr.tensor_idx) ptr.multi_dim_idx

/-- Get the scalar value at an input pointer -/
def getValueAtPointer (input : MultiTensorInput) (ptr : InputPointer input) : ℝ :=
  (input.tensors ptr.tensor_idx) ptr.multi_dim_idx

/-- Generalized dependency mapping: maps output indices to lists of k input pointers -/
structure GeneralizedDependencyMapping (input : MultiTensorInput) (output_dims : List ℕ) (k : ℕ) where
  map : Index output_dims.length → List.Vector (InputPointer input) k
  -- All mapped pointers must be valid
  valid : ∀ (out_idx : Index output_dims.length),
    validIndex output_dims out_idx →
    ∀ i : Fin k, validInputPointer input ((map out_idx).get i)

/-- Multi-tensor SIMD function structure -/
structure SIMDFunction (input : MultiTensorInput) (output_dims : List ℕ) where
  k : ℕ  -- Number of scalar inputs for kernel function
  kernel : KernelFunction k  -- Reuse kernel function from original definition
  dependency : GeneralizedDependencyMapping input output_dims k

/-- Apply multi-tensor SIMD function to compute output tensor element at given index -/
def applySIMDAt (input : MultiTensorInput) (output_dims : List ℕ)
    (simd : SIMDFunction input output_dims)
    (output_idx : Index output_dims.length) : ℝ :=
  let input_pointers := simd.dependency.map output_idx
  let input_scalars := List.Vector.map (getValueAtPointer input) input_pointers
  simd.kernel input_scalars

/-- Complete multi-tensor SIMD function application -/
def applySIMD (input : MultiTensorInput) (output_dims : List ℕ)
    (simd : SIMDFunction input output_dims) : Tensor output_dims :=
  fun output_idx => applySIMDAt input output_dims simd output_idx

/-- Predicate to check if a function is a multi-tensor SIMD function -/
def isSIMDFunction (input : MultiTensorInput) (output_dims : List ℕ)
    (f : MultiTensorInput → Tensor output_dims) : Prop :=
  ∃ (simd : SIMDFunction input output_dims),
    f input = applySIMD input output_dims simd

/-- Mathematical representation: Y[i_out] = θ((X_ptr.tensor_idx[ptr.multi_dim_idx])_ptr∈τ(i_out)) -/
theorem multi_tensor_simd_characterization (input : MultiTensorInput) (output_dims : List ℕ)
    (simd : SIMDFunction input output_dims)
    (output_idx : Index output_dims.length) :
    (applySIMD input output_dims simd) output_idx =
    applySIMDAt input output_dims simd output_idx := by
  rfl

/-- Well-formed kernel function definition -/
def WellFormedKernel (k : ℕ) (θ : KernelFunction k) : Prop :=
  ∀ i : Fin k, ∃ (v v' : List.Vector ℝ k),
    (v.get i ≠ v'.get i) ∧
    (∀ j : Fin k, j ≠ i → v.get j = v'.get j) ∧
    (θ v ≠ θ v')

/-- k-ary addition kernel function: sums all k input elements -/
def sumKernel (k : ℕ) : KernelFunction k :=
  fun v => ∑ j : Fin k, v.get j

/-- Matrix multiplication kernel function: computes dot product of k pairs of values -/
def matmulKernel (k : ℕ) : KernelFunction (2 * k) :=
  fun v => ∑ i : Fin k, (v.get ⟨i, by omega⟩) * (v.get ⟨k + i, by omega⟩)

/-- Theorem: k-ary addition is a well-formed kernel function -/
theorem sum_is_wellformed (k : ℕ) : WellFormedKernel k (sumKernel k) := by
  intro i
  -- 构造两个向量：一个在位置i处为0，其他为1；另一个全为1
  use (List.Vector.ofFn (fun j => if j = i then 0 else 1)), (List.Vector.ofFn (fun _ => 1))
  -- 证明的关键：这个构造满足well-formed的三个条件
  constructor
  · -- 1. v.get i ≠ v'.get i
    simp [List.Vector.get_ofFn]
  · constructor
    · -- 2. Prove ∀ j : Fin k, j ≠ i → v.get j = v'.get j
      intro j hij
      simp [List.Vector.get_ofFn, hij]
    · -- 3. Prove sumKernel k v ≠ sumKernel k v'
      simp only [sumKernel]
      -- Goal: sumKernel k v ≠ sumKernel k v'
      -- We'll prove the sums are k-1 and k respectively.
      by_cases h : k = 0
      · subst h; exact Fin.elim0 i
      · have k_pos : 0 < k := Nat.pos_of_ne_zero h

        -- Prove sum for v = List.Vector.ofFn (fun j => if j=i then 0 else 1)
        let v := List.Vector.ofFn (fun j => if j=i then (0:ℝ) else 1)
        have sum_v : ∑ j, v.get j = (k:ℝ) - 1 := by
          -- Rewrite using get_ofFn
          simp_rw [v, List.Vector.get_ofFn]
          -- Now we have: ∑ j, if j = i then 0 else 1 = k - 1
          -- The sum equals the number of j ≠ i, which is k - 1
          have h1 : (∑ j : Fin k, if j = i then (0:ℝ) else 1) =
                    ∑ j : Fin k, if j ≠ i then 1 else 0 := by
            congr 1; ext j
            by_cases h : j = i <;> simp [h]
          rw [h1]
          rw [Finset.sum_boole]
          -- The filter gives us all elements except i
          have h2 : Finset.filter (fun x => ¬x = i) Finset.univ = Finset.univ.erase i := by
            ext j
            simp [Finset.mem_filter, Finset.mem_erase]
          rw [h2]
          simp [Finset.card_erase_of_mem (Finset.mem_univ i),
                Finset.card_univ, Fintype.card_fin]
          rw [Nat.cast_sub (by linarith : 1 ≤ k), Nat.cast_one]

        -- Prove sum for v' = List.Vector.ofFn (fun _ => 1)
        let v' := List.Vector.ofFn (fun (_ : Fin k) => (1:ℝ))
        have sum_v' : ∑ j, v'.get j = (k:ℝ) := by
          simp_rw [v', List.Vector.get_ofFn]
          simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]

        rw [sum_v, sum_v']
        norm_num
end SIMD
