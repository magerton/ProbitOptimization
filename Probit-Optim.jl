using DataFrames
using Distributions
using NLopt
using Optim
using Dates

# projectDir  = "~/OneDrive/Rice/Class/econ 515 - Labor/NLopt/"
# projectDir  = "C:/mja3/SkyDrive/Rice/Class/econ 515 - Labor/NLopt/"
projectDir  = "C:/Users/magerton/OneDrive/Rice/Class/econ 515 - Labor/NLopt/"
cd(projectDir)

# --------------------------------------------------------------
# ------------------------ Logging    --------------------------
# --------------------------------------------------------------

doLog = false
originalSTDOUT = STDOUT

if doLog == true 
  (outRead, outWrite) = redirect_stdout()
  println("-------------------------------------------------------------------")
  println("Results for Homework 5\nStarted run at " * string(now()))
  println("-------------------------------------------------------------------")
end

# ------------------------------------------------------------
# ------------------------ Probit DGP ------------------------
# ------------------------------------------------------------


N     = 10^5
k     = 4
u     = rand(Normal(),N)
X     = [ones(N) reshape( rand(Uniform(-1,1), N*(k-1)), (N,k-1) )]
βTrue = [2.0, 3, 5, 10]
y     = X*βTrue + u
d     = 1.0 * (y .>= 0 )


include("Probit-Functions.jl")

# Being declared in main prog makes these global
maxit = 100
count = 0

# ------------------------------------------------------------
# -------------- Optim Optimization --------------------------
# ------------------------------------------------------------

for meth in [:bfgs :l_bfgs :nelder_mead :newton :simulated_annealing]
  println("\n------------- doing method $meth ---------------\n\n")
  tic();  global count = 0  
  res = Optim.optimize(LL, g!, h!, zeros(k), method=meth, iterations=maxit )
  println("\n$(count) evaluations over $(toq()) seconds \n\n $(res)") 
end

res   = Optim.optimize(LL, g!, h!, zeros(k), method=:newton, iterations=maxit )
vcv   = vcov(res, h!)
myse  = se(res, h!)
tstat = coef(res) ./ se(res, h!)
pval  = cdf(Normal(), -abs(tstat))

# ------------------------------------------------------------
# -------------- NLopt Optimization --------------------------
# ------------------------------------------------------------

# Note that GlobalocalOpt routines require a local optimzation routine
# Also, global optimizers (usually) require lower and upper bounds
# The sets ending in "Constr" seem to be okay w/ nonlinear constraints

# Global routines
meth_Global         = [ :GN_DIRECT, :GN_DIRECT_L, :GN_DIRECT_L_RAND, :GN_DIRECT_NOSCAL, 
                        :GN_DIRECT_L_NOSCAL, :GN_DIRECT_L_RAND_NOSCAL, :GN_ORIG_DIRECT, 
                        :GN_ORIG_DIRECT_L, :GN_ISRES, :GN_ESCH, :GN_CRS2_LM  ]
meth_GlobalBroken   = [:GD_STOGO, :GD_STOGO_RAND]
meth_GlobalLocalOpt = [:GN_MLSL_LDS, :GN_MLSL, :G_MLSL, :G_MLSL_LDS, :GD_MLSL_LDS, 
                       :GD_MLSL, :AUGLAG, :AUGLAG_EQ]

# Derivative-based routines
methDerivConstr = [:LD_MMA , :LD_SLSQP, :LD_CCSAQ ]
methDeriv       = [:LD_LBFGS, :LD_VAR1, :LD_VAR2, :LD_TNEWTON, :LD_TNEWTON_RESTART, 
                   :LD_TNEWTON_PRECOND, :LD_TNEWTON_PRECOND_RESTART ]
methDerivBroken = [:LD_LBFGS_NOCEDAL]

# Non derivative-based routines
methNonDerivBroken = [:LN_NEWUOA]
methNonDeriv       = [:LN_PRAXIS, :LN_NEWUOA_BOUND, :LN_NELDERMEAD, :LN_SBPLX, :LN_BOBYQA]
methNonDerivConstr = [:LN_COBYLA, :LN_AUGLAG, :LD_AUGLAG, :LN_AUGLAG_EQ, :LD_AUGLAG_EQ ]


for meth in [meth_Global, methNonDeriv, methNonDerivConstr, methDeriv, methDerivConstr]

  println("\n------------- doing method $meth ---------------\n\n")
  tic();  global count = 0  

  opt = NLopt.Opt(meth, k)
  
  NLopt.lower_bounds!(opt, -100*ones(k))
  NLopt.upper_bounds!(opt,  100*ones(k))
  NLopt.xtol_rel!(opt,1e-12)
  NLopt.maxeval!(opt, maxit)

  # Configure local optimization if needed
  if any( meth .== meth_GlobalLocalOpt ) 
    opt_local = Opt(:LN_NELDERMEAD, 2)
    local_optimizer!(opt, opt_local)
  end

  NLopt.min_objective!(opt, LL)

  ### this is not used but would be good to figure out
  # inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
  # inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

  (minf,minx,ret) = NLopt.optimize(opt, zeros(k))
  println("\n$(count) evaluations over $(toq()) seconds. Returned $ret") 
  println("\t\tvalue = $(round(minf,5)) at $(round(minx,3))")
end

# --------------------------------------------------------------
# ------------------------ Logging    --------------------------
# --------------------------------------------------------------

if doLog == true

  println("-------------------------------------------------------------------")
  println("Finished run at " * string(now()))
  println("-------------------------------------------------------------------")

  close(outWrite)
  stringOut = readavailable(outRead)
  close(outRead)
  redirect_stdout(originalSTDOUT)

  f = open("Probit-Results.txt", "w")
  write(f, stringOut )
  close(f)

  println(stringOut)
end
