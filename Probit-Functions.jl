# -------------- Normal distribution functions prevent 0.0 and 1.0 ----------------------

function normcdf(x::Vector{Float64})
  out = Distributions.cdf(Distributions.Normal(), x)
  out + (out .== 0.0)*eps(1.0) - (out .== 1.0)*eps(1.0)
end


function normpdf(x::Vector{Float64})
  Distributions.pdf(Distributions.Normal(), x)
end

# -------------- Likelihood, gradietn and hessian ----------------------
# see q/lambda trick in Greene chapter on Models for discrete choice!

function λ(β::Vector{Float64})
	q =	2d-1
	q .* normpdf(q .* X*β) ./ normcdf(q.*X*β)
end


function LL(β::Vector{Float64})
	out = - sum( log( normcdf( (2d-1) .* X*β) ) )
	countPlus!(out)
	return(out)
end


function LL(β::Vector{Float64}, grad::Vector{Float64})
	out = - sum( log( normcdf( (2d-1) .* X*β) ) )
	if length(grad) > 0
		grad[:] = - sum( λ(β) .* X,  1 )
	end
	countPlus!(out)		
	return(out)
end



function g!(β::Vector{Float64}, grad::Vector{Float64})
  grad[:] = - sum( λ(β) .* X,  1 )
end


function h!(β::Vector{Float64}, hess::Matrix{Float64})
  hh = zeros(size(hess))
  A = λ(β) .* ( λ(β) + X*β )
  
  for i in 1:size(X)[1]
    hh += A[i] * X[i,:]'*X[i,:]
  end
  
  hess[:] = hh
end


# -------------- Results ----------------------

function vcov(obj::Optim.OptimizationResults, h!)
	β    = obj.minimum
	k    = length(β)
	N    = maximum(size(X))
	hess = zeros((k,k))
	h!(β, hess)
	N*(hess \ eye(k))
end

function se(obj::Optim.OptimizationResults, h!)
	sqrt(diag(vcov(obj, h!)))
end

function coef(obj::Optim.OptimizationResults)
	obj.minimum
end

# -------------- Convenience ----------------------

function printCounter(count)
	if count <= 5
		denom = 1
	elseif count <= 50
		denom = 10
	elseif count <= 200
		denom = 25
	elseif count <= 500
		denom = 50
	elseif count <= 2000
		denom = 100
	else
		denom = 500
	end
	mod(count, denom) == 0 
end


function countPlus!()
  global count += 1
  if printCounter(count) 
    println("Eval $(count)")
  end
end


function countPlus!(out::Float64)
  global count += 1
  if printCounter(count) 
    println("Eval $(count): value = $(round(out,5))")
  end
    return count
end
