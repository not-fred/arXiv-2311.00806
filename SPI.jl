"""
    julia SPI.jl d₁ d₂ K₁ K₂ [file]

Script used in [arXiv:2311.00806](https://arxiv.org/abs/2311.00806) for
maximising the score over pure product states of a j₁ ⊗ j₂ spin pair,
to find a lower bound for the separable bound for a precession-based
entanglement witness. See the paper for more detail.

# Arguments
- `d₁::Integer`: Dimension d₁ = 2j₁ + 1 of the first spin system
- `d₂::Integer`: Dimension d₁ = 2j₂ + 1 of the first spin system
- `K₁::Integer`: Minimum (odd) number of measurement settings
- `K₂::Integer`: Maximum (odd) number of measurement settings
- `file::String`: Data will be saved in $file-K$K.csv. Returns to standard output if none specified

# Package versions
For the reported results, version 1.6.6 of `julia` was used, with the packages
- `WignerSymbols`: v2.0.0
- `Memoization`: v0.2.1
- `ArnoldiMethod`: v0.2.0
"""

dArr  = parse(Int,ARGS[1]):parse(Int,ARGS[2])
Karr  = parse(Int,ARGS[3]):2:parse(Int,ARGS[4])
file  = length(ARGS) ≥ 5 ? ARGS[5] : false

# Tolerence for SPI solver to stop
ε = 1E-5

# Number of threads for parallel computation
numThreads = Threads.nthreads()

using LinearAlgebra, SparseArrays, Dates, DelimitedFiles
using WignerSymbols, Memoization, ArnoldiMethod

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


"""
    SepEig(A,d₁,d₂;tol=1E-7,maxiter=10*d₁^2*d₂^2)

Seperable power iteration algorithm to heuristically find the maximum value of
(⟨ψ₁|⊗⟨ψ₂|)A(|ψ₁⟩⊗|ψ₂⟩) over separable pure states |ψ₁⟩⊗|ψ₂⟩. See Appendix D1
of [arXiv:2311.00806](https://arxiv.org/abs/2311.00806) for more information
and local convergence guarantees
"""
function SepEig(A,d₁,d₂;tol=1E-7,maxiter=10*d₁^2*d₂^2)
    maxA = 0
    for j ∈ 1:d₁, k ∈ 1:d₁, l ∈ 1:2
        # We choose starting vectors |ψ₁⟩ such that the set {|ψ₁⟩⟨ψ₁|} spans the
        # operator space. In particular, we choose the eigenvectors of the Gell-Mann
        # matrices, with some randomly-chosen vectors thrown into the mix
        v₁₂ = [j == k && l == 2 ? sparse(normalize(rand(d₁)+im*rand(d₁))) : spzeros(ComplexF64,d₁), spzeros(ComplexF64,d₂)]
        λ₁₂ = [Inf,0.0]
        if j == k && l ≠ 2
            v₁₂[1][j] = 1
        else
            v₁₂[1][j] = 1/√2
            v₁₂[1][k] = (j<k ? 1 : -1)*(l==1 ? 1im : 1)/√2
        end
        
        # We perform the power iteration until the tolerence is met,
        # or the maximum number of iterations is performed
        N = 0
        while abs(diff(λ₁₂)[1]) > tol && N < maxiter
            tr₂A = kron(sparse(v₁₂[1]'),I(d₂))*A*kron(v₁₂[1],I(d₂))
            try
                # partialschur is faster for sparse matrices
                # for finding the next interation |b⟩
                sch,_ = partialschur(tr₂A, nev=1, which=LR())
                v₁₂[2] = sch.Q[:]
                λ₁₂[2] = sch.eigenvalues[end] |> real
            catch
                # Fallback to standard eigensolver if partialschur fails
                eigλ,eigV = eigen(collect(tr₂A+tr₂A')/2)
                v₁₂[2] = sparse(eigV[:,end])
                λ₁₂[2] = eigλ[end] |> real
            end
            tr₁A = kron(I(d₁),sparse(v₁₂[2]'))*A*kron(I(d₁),v₁₂[2])
            try
                sch,_ = partialschur(tr₁A, nev=1, which=LR())
                v₁₂[1] = sch.Q[:]
                λ₁₂[1] = sch.eigenvalues[end] |> real
            catch
                eigλ,eigV = eigen(collect(tr₁A+tr₁A')/2)
                v₁₂[1] = sparse(eigV[:,end])
                λ₁₂[1] = eigλ[end] |> real
            end
            N += 1
        end
        abs(diff(λ₁₂)[1]) > tol && println("Didn't converge")
        currA = sum(λ₁₂)/2
        if maxA < currA
            maxA = currA
        end
    end
    return maxA
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
            # Generate ΔQ_K
            ΔQKarr = genΔQK(Karr,(d₁-1)//2,(d₂-1)//2)
            
            # Loop through each number of measurements K
            Threads.@threads for iK in 1:length(Karr)
                if Parr[j,k,iK] > 0
                    # Skip if already calculated
                    continue
                end
                K = Karr[iK]
                if d₁+d₂ < K+2
                    # Known to be trivial when j₁ + j₂ < K/2
                    Parr[j,k,iK] = 1/2
                else
                    # Get the lower bound of the solution
                    Parr[j,k,iK] = 1/2 + SepEig(ΔQKarr[iK]/2,d₁,d₂,tol=ε)
                end
                outputStrings[j,k,iK] = "$d₁,$d₂,$K,$(Parr[j,k,iK])"
                if file != false
                    open("$file-K$K.csv","a") do io
                        println(io,outputStrings[j,k,iK])
                    end
                end
            end
        end
        println("End time: $(now()), time taken: $(now()-startTime)")
        println("")
    end
end

# If no file was specified, print out the results
if file == false
    println(join([outputStrings[j,k] for j ∈ 1:length(outputStrings[:,1]), k ∈ 1:length(outputStrings[1,:]) if k ≥ j],"\n"))
end
