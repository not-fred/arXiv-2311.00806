"""
    julia SDP.jl d₁ d₂ K₁ K₂ [file]

Script used in [arXiv:2311.00806](https://arxiv.org/abs/2311.00806) for
maximising the score over positive partial transpose states of a j₁ ⊗ j₂
spin pair, to find the upper bound of the separable bound for a precession-
based entanglement witness. See the paper for more detail.

# Arguments
- `d₁::Integer`: Dimension d₁ = 2j₁ + 1 of the first spin system
- `d₂::Integer`: Dimension d₁ = 2j₂ + 1 of the first spin system
- `K₁::Integer`: Minimum (odd) number of measurement settings
- `K₂::Integer`: Maximum (odd) number of measurement settings
- `file::String`: Data will be saved in $file-K$K.csv. Returns to standard output if none specified

# Package versions
For the reported results, version 1.6.6 of `julia` was used, with the packages
- `JuMP`: v1.12.0
- `COSMO`: v0.8.8
- `SCS`: v1.1.2
- `WignerSymbols`: v2.0.0
- `Memoization`: v0.2.1
"""

dArr  = parse(Int,ARGS[1]):parse(Int,ARGS[2])
Karr  = parse(Int,ARGS[3]):2:parse(Int,ARGS[4])
file  = length(ARGS) ≥ 5 ? ARGS[5] : false

dArr = 4:5
Karr = [3]

# Tolerence for SDP solver to stop
ε = 1E-5

# Number of threads for parallel computation
numThreads = Threads.nthreads()

# Standard libraries
using LinearAlgebra, SparseArrays, Dates, DelimitedFiles

# Other libraries
using JuMP, COSMO, SCS, WignerSymbols, Memoization
import Convex.partialtranspose

"""
    binCoef(x)

Returns 2⁻ˣ × choose(x,⌊x/2⌋). For `x` ≤ 62, the exact value is returned.
For x > 62, a series where log(binCoef(x)) is of order O(1/x²¹) is used.
"""
function binCoef(x)
    x = 2*(x÷2)
    if x ≤ 62
        return binomial(BigInt(x),BigInt(x÷2))/2^x
    else
        # We use 2⁻ˣ × choose(x,⌊x/2⌋) = exp(f(x) + O(1/x²¹)) × √(2/(π*x))
        x = BigFloat(x)
        fx = exp(
                -1/4x + 1/24x^3 - 1/20x^5 + 17/112x^7 - 31/36x^9 +
                691/88x^11 - 5461/52x^13 + 929569/480x^15 - 3202291/68x^17 + 221930581/152x^19
            )
        return fx/√(π*x/2)
    end
end

"""
    pTranspose(ρ,d₁::Int=Int(√size(ρ)[1]),d₂::Int=Int(√size(ρ)[1]))

Returns the partial transpose of `ρ`, defined by ⟨n₁,n₂|ρᵀ²|m₁,m₂⟩ = ⟨n₁,m₂|ρ|m₁,n₂⟩
"""
function pTranspose(ρ,d₁::Int=Int(√size(ρ)[1]),d₂::Int=Int(√size(ρ)[1]))
    ρᵀ² = copy(ρ)*0
    for n₁ ∈ 0:d₁-1, n₂ ∈ 0:d₂-1, m₁ ∈ 0:d₁-1, m₂ ∈ 0:d₂-1
        # ⟨n₁,n₂|ρᵀ²|m₁,m₂⟩ = ⟨n₁,m₂|ρ|m₁,n₂⟩
        (ishermitian(ρᵀ²) ? ρᵀ².data : ρᵀ²)[n₁*d₂ + n₂ + 1, m₁*d₂ + m₂ + 1] = ρ[n₁*d₂ + m₂ + 1, m₁*d₂ + n₂ + 1]
    end
    return ρᵀ²
end

"""
    sgnJx(j,m₁,m₂)

Returns ⟨j,m₁|sgn(Jₓ)|j,m₂⟩; see Eq. (B5) of [arXiv:2311.00806](https://arxiv.org/abs/2311.00806)
"""
@memoize function sgnJx(j,m₁,m₂)
    if (m₁-m₂)%2 == 0
        return 0
    else
        return (-1)^((m₂-m₁-1)÷2)*√(
            (j+m₁)^((j+m₁)%2)*
            (j-m₁)^((j-m₁)%2)*
            (j+m₂)^((j+m₂)%2)*
            (j-m₂)^((j-m₂)%2)*
            binCoef(j+m₁)*binCoef(j-m₁)*
            binCoef(j+m₂)*binCoef(j-m₂)
        )/(m₂-m₁)
    end
end

"""
    genΔQK(Karr,j₁,j₂)

Returns an array [ΔQ_K₁, ΔQ_K₂, …] for Kₗ ∈ Karr,
where Q_K = ∑ₖ pos[Jₖ]/K and ΔQ_K := 2Q_K − 𝟙;
see Eq. (3) of [arXiv:2311.00806](https://arxiv.org/abs/2311.00806).
ΔQ_K is returned in the product basis of j₁ ⊗ j₂
"""
function genΔQK(Karr,j₁,j₂)
    # From addition of angular momentum, |j₁-j₂| ≤ j ≤ j₁+j₂,
    # where j is the spin associated with the irreducible block
    jArr = abs(j₂-j₁):(j₂+j₁)
    
    # Get the dimensions from the spin arguments
    d₁ = Int(2j₁+1)
    d₂ = Int(2j₂+1)
    
    # Cummulative dimensions of the block sum
    # If index 0+1 to d₁ is the first block, d₁+1 to d₂ is
    # the second block, etc., then this array contains
    # [0,d₁,d₂,d₃,…]
    cdArr = [0;cumsum(2jArr .+ 1)[1:end-1]]
    
    # Unitary to be populated with the Clebsch–Gordan coefficients
    U = spzeros(d₁*d₂,d₁*d₂)
    
    # Array that contains Q_K, in the block-diagonal basis
    ΔQKarr = [spzeros(d₁*d₂,d₁*d₂) for _ in Karr]
    
    # Loop through all blocks
    for iJ ∈ 1:length(jArr)
        Δ = cdArr[iJ] + 1 # Get the index where this block starts
        j = jArr[iJ] # Spin of current block
        for mⱼ ∈ -j:j # Loop through all |j,mⱼ⟩
            iMⱼ = Int(Δ+mⱼ+j) # Index of |j,mⱼ⟩
            for iK in 1:length(Karr)
                K = Karr[iK]
                # Loop through |j,mₖ⟩, according to block diagonal condition: see
                # Eq. (B2) & (B3) of [arXiv:2311.00806](https://arxiv.org/abs/2311.00806)
                for mₖ ∈ mⱼ+K:2K:j
                    iMₖ = Int(Δ+mₖ+j)
                    ΔQKarr[iK][iMⱼ,iMₖ] = ΔQKarr[iK][iMₖ,iMⱼ] = sgnJx(j,mⱼ,mₖ)
                end
            end
            # Populate the Clebsch—Gordan coefficients
            for m₁ ∈ max(-j₁,mⱼ-j₂):min(j₁,mⱼ+j₂)
                U[Int((j₁+m₁)*d₂ + (j₂+mⱼ-m₁) + 1),iMⱼ] = clebschgordan(j₁,m₁,j₂,mⱼ-m₁,j)
            end
        end
    end
    # Return QK in the product basis of j₁ and j₂
    return [sparse(Hermitian(Float64.(U*ΔQK*U'))) for ΔQK in ΔQKarr]
end

# Generate array to contain separable bounds and output strings
Parr = zeros(length(dArr),length(dArr),length(Karr))
outputStrings = ["" for _ in 1:length(dArr), _ in 1:length(dArr), _ in 1:length(Karr)]

# Load data if partial computation was previously done
for iK in 1:length(Karr)
    K = Karr[iK]
    try
        dat = readdlm("$file-K$K.csv",',')
        for j ∈ 1:length(dat[:,1])
            cd₁,cd₂,cK,cP = dat[j,:]
            if cK == K
                iD₁ = indexin(cd₁,dArr)[1]
                iD₂ = indexin(cd₂,dArr)[1]
                # If data already exists, copy into array
                if !isnothing(iD₁) && !isnothing(iD₂)
                    Parr[iD₁,iD₂,iK] = cP
                end
            end
        end
    catch
    end
end

# Loop through all j₂ ≥ j₁
for j ∈ 1:length(dArr)
    for k ∈ j:length(dArr)
        if all(Parr[j,k,:] .> 0)
            # Skip this (d₁,d₂) pair if separable bound has been calculated for all values of K
            continue
        end
        startTime = now()
        println("$j of $(length(dArr)), $(k-j+1) of $(length(dArr)-j+1)")
        d₁ = dArr[j]
        d₂ = dArr[k]
        if d₁+d₂ < minimum(Karr) + 2
            # Known to be trivial when j₁ + j₂ < K/2
            Parr[j,k,:] .= 1/2
        else
            println("Generating Observables…")
            ΔQKarr = genΔQK(Karr,(d₁-1)//2,(d₂-1)//2)

            println("Starting SDP…")
            try
                # Use COSMO solver to solve the SDP
                model = Model(COSMO.Optimizer)
                
                # Constraint ρ to be a density operator: ρ ⪰ 0 and tr(ρ) = 1
                @variable(model, ρ[1:d₁*d₂,1:d₁*d₂] ∈ HermitianPSDCone())
                @constraint(model, tr(ρ)==1)
                
                # Partial transpose constraint: ρᵀ² = ρ
                @constraint(model, Hermitian(pTranspose(ρ,d₁,d₂)) ∈ HermitianPSDCone())
                
                # Loop through each number of measurements K
                for iK in 1:length(Karr)
                    if Parr[j,k,iK] > 0
                        # Skip if already calculated
                        continue
                    end
                    K = Karr[iK]
                    if d₁+d₂ < K+2
                        # Known to be trivial when j₁ + j₂ < K/2
                        Parr[j,k,iK] = 1/2
                    else
                        # Maximise tr(ρ ΔQ_K/2)
                        @objective(model, Max, real(dot(ΔQKarr[iK]/2,ρ)))
                        MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 50_000)
                        MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), ε)
                        MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), ε)
                        optimize!(model)
                        if max(dual_objective_value(model),objective_value(model)) > 1
                            # It must be a probability, so it obviously
                            # didn’t converge if it’s not
                            throw("COSMO didn’t converge")
                        end
                        # Get the upper bound of the solution
                        Parr[j,k,iK] = max(dual_objective_value(model),objective_value(model)) + 1/2
                    end
                    outputStrings[j,k,iK] = "$d₁,$d₂,$K,$(Parr[j,k,iK])"
                    if file != false
                        open("$file-K$K.csv","a") do io
                            println(io,outputStrings[j,k,iK])
                        end
                    end
                end
            catch
                # COSMO fails sometimes, in which case we fallback onto SCS
                model = Model(SCS.Optimizer)
                @variable(model, ρ[1:d₁*d₂,1:d₁*d₂] ∈ HermitianPSDCone())
                @constraint(model, tr(ρ)==1)
                @constraint(model, Hermitian(pTranspose(ρ,d₁,d₂)) ∈ HermitianPSDCone())
                for iK in 1:length(Karr)
                    if Parr[j,k,iK] > 0
                        continue
                    end
                    K = Karr[iK]
                    if d₁+d₂ < K+2
                        Parr[j,k,iK] = 1/2
                    else
                        @objective(model, Max, real(dot(ΔQKarr[iK]/2,ρ)))
                        MOI.set(model, MOI.RawOptimizerAttribute("max_iters"), 50_000)
                        MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), ε)
                        MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), ε)
                        optimize!(model)
                        Parr[j,k,iK] = max(dual_objective_value(model),objective_value(model)) + 1/2
                    end
                    outputStrings[j,k,iK] = "$d₁,$d₂,$K,$(Parr[j,k,iK])"
                    if file != false
                        open("$file-K$K.csv","a") do io
                            println(io,outputStrings[j,k,iK])
                        end
                    end
                end
            end
        end
        println("$j of $(length(dArr)), $(k-j+1) of $(length(dArr)-j+1) end at time: $(now()), time taken: $(now()-startTime)")
        println("")
    end
end

# If no file was specified, print out the results
if file == false
    println(join([outputStrings[j,k] for j ∈ 1:length(outputStrings[:,1]), k ∈ 1:length(outputStrings[1,:]) if k ≥ j],"\n"))
end
