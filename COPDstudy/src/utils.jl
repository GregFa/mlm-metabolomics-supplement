#=
Synopsosis: `utils.jl`.

Functions list:


- tstat2pval:
    Returns p-values from the t-statistics and degree of freedom
    
=#

"""
**tstat2pval** -*Function*

    tstat2pval(mTstats::AbstractMatrix, df::Int64; istwotailed = true) => Matrix

Returns p-values from the t-statistics and degree of freedom.

***Arguments***

- `mTstats` vector.
- `df` degree of freedom.
- `istwotailed` indicate if two-tailed test or one tail test.
"""
function tstat2pval(mTstats::AbstractMatrix, df::Int64; istwotailed = true)
        
    mTstats = permutedims(mTstats)
    
    # initialize pValue matrix
    q, p = size(mTstats) # size of B' 
    pVals = zeros(q, p)
    
    if istwotailed
       a = 2
    else
        a = 1
    end

    for i in 1:p
       pVals[:,i] = ccdf(TDist(df), abs.(mTstats[:,i])).*a
    end
    
    return permutedims(pVals)
end


"""
**pval2qval** -*Function*

    pval2qval(mTstats::AbstractMatrix, df::Int64; istwotailed = true) => Matrix

Returns q-values from the p-values.

"""
function pval2qval(vPvals::AbstractVector)
        
    # Step 1: Estimate proportion of truly null hypothesis
    n = length(vPvals)
    Π₀ = estimate(vPvals, StoreyBootstrap())
    
    # Step 2: sort original p-values
    sort!(vPvals)

    # Step 3: calculate the q-value for the largest p-value
    vQvals = copy(vPvals)
    vQvals[end] = vQvals[end]*Π₀

    # Step 4: calculate the rest of the q-values such as
    for i in n-1:-1:1
       vQvals[i] = minimum([(Π₀*n*vPvals[i])/i, vQvals[i+1]]) 
    end
    
    return vQvals
end


function pval2qval2(vPvals::AbstractVector)
    
    # Step 1: Estimate proportion of truly null hypothesis
    n = length(vPvals)
    Π₀ = estimate(vPvals, StoreyBootstrap())
    
    vQvals = adjust(vPvals, BenjaminiHochbergAdaptive(Π₀))
        
    return vQvals
end




function pval2qval(mPvals::AbstractMatrix)
    
    mPvals = permutedims(mPvals)
    
    # initialize mQvals matrix
    q, p = size(mPvals) # size of B' 
    mQvals = zeros(q, p)
    
    
    
    for i in 1:p
       mQvals[:,i] = pval2qval2(mPvals[:, i])
    end
    
    return permutedims(mQvals)
end
