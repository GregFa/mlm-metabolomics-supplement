### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 35fab8f4-a186-4d05-8369-91f574f9aa4c
begin
	#############
	# Libraries #
	#############
	using CSV, DataFrames, DataFramesMeta, Missings, CategoricalArrays
	using StatsBase, Statistics, MultivariateStats, MatrixLM
	using Random, Distributions, StatsModels
	using LinearAlgebra, PrettyTables
	using FreqTables, Plots, StatsPlots, PlutoUI, Images, FileIO
	using Plots.PlotMeasures


	######################
	# External functions #
	######################
	include("../../src/mLinearModel.jl");
end;

# ╔═╡ 0b37e078-ce99-4268-9749-9933735129c3
md"""
# Lipidomics Analysis Using Matrix Linear Models  
---
### Gregory Farage, Saunak Sen *et al.*
    gfarage@uthsc.edu / sen@uthsc.edu
    Division of Biostatistics
    Department of Preventive Medicine
    University of Tennessee Health Science Center
    Memphis, TN

## Background

For decades, statins have been effective and widely popular cholesterol-lowering agents with substantial benefits for preventing and treating cardiovascular disease.
However, some patients can not tolerate statins. Statin intolerance is usually associated with muscle pain as side effects, also known as statin-associated muscle symptoms (SAMS). 

**SAMS are particularly difficult to treat:**
* no validated biomarkers or tests to confirm patient self-reports of SAMS
* some self-reported patients have non-specific muscle pain not attributable to statin therapy


## Project Summary

This project seeks to identify biomarkers and pathway's related to susceptibility to SAMS based on metabolomic and lipidomic studies.   

**Comparisons of metabolomic/lipidomic profiling are made for:**
* CS-baseline group vs CS group (paired samples)
* CN group vs CS group (unpaired samples)

**Group’s acronym definition:**
* CS-baseline group: Patients with documented SAMS off statin    

* CS group: Patients with documented SAMS following rechallenge with statin (Rechallenge) for up to 4 weeks or until symptomatic.     
       
* CN group: Controls patients who have no history of SAMS and who are currently treated with statin (on statin).
"""

# ╔═╡ 20574596-5e33-4be5-9958-e688fa712d84
PlutoUI.TableOfContents()

# ╔═╡ 51459290-6b57-4ec7-8750-33e8014e7f42
md"""
## Data set glimpse
"""

# ╔═╡ 2b7def0b-703c-433a-857b-f83b8986f312
begin
#############
# Load data #
#############

# Load look up table for lipids
LipidsXref = realpath((@__DIR__)*"/../../data/data_processed/inl2b_Lipids_Xref.csv")
dfLipidsXref = DataFrame(CSV.File(LipidsXref));

# Load data set
posLipids = realpath((@__DIR__)*"/../../data/data_processed/inl2b_Lipids.csv")
dfLipids = DataFrame(CSV.File(posLipids));

################
# Filter cases #
################
# true => unpaired
# false => paired (rechallenge)
pairingFlag = true;
df = getCases(dfLipids, isunpaired = pairingFlag);

# Check to filter none fish oil users 
hasFishOil = true;

if !hasFishOil
    # Get only individuals with no fish oil
    filter!(row-> row.FishOil == "no" ,df);
end


# standardize lipids mass-to-charge ratio
funStandardize!(df, isunpaired = pairingFlag);
end;

# ╔═╡ 536cf374-6611-4a71-8842-719e50300633
md"""
Display the first variables:
"""

# ╔═╡ a400ae63-60fe-4a98-b58a-f519edfc3644
begin
# display first samples
# mytf = tf_html_default
# pretty_table(Matrix(df[1:7, 1:8]), names(df)[1:8]; backend = :html, tf = mytf)
df[1:7, 1:8]
end

# ╔═╡ 3b95f87e-6abe-4935-86ac-6598b5ee04bd
md"""
## Individual characteristics
"""

# ╔═╡ 2c7a26f6-cfba-48d9-a30a-22a0d1ef43ce
md"""
Two-way contingency tables considering `Group` and `FishOil` variables: 
"""

# ╔═╡ 67743f5d-454a-4be2-87fb-ac9e5062a059
begin
	fullTable = freqtable(dfLipids[:, [:FishOil, :Group]], :FishOil, :Group)
matFullfreq = collect(fullTable[:,:]);
matFullfreq = vcat(matFullfreq, sum(matFullfreq, dims= 1))
matFullfreq = hcat(matFullfreq, sum(matFullfreq, dims= 2))
dfFreqFull = DataFrame(matFullfreq, :auto)
rename!(dfFreqFull, Symbol.(vcat(names(fullTable,2), "Total")))
insertcols!(dfFreqFull, 1, :_ => vcat("Fish Oil: ".*(names(fullTable,1)), Symbol("Total")))

# indivTable = insertcols!(dfFreqFull, 1, :_ => vcat("Fish Oil: ".*(names(fullTable,1)), Symbol("Total")));
# hl_v = HTMLHighlighter((data, i, j) -> (j == 1), HTMLDecoration(font_weight = "bold"));
# pretty_table(Matrix(indivTable[:, :]), names(indivTable);highlighters = hl_v, backend = :html, tf = mytf)
end

# ╔═╡ f5f01515-175f-4684-9758-512b709b4687
md"""
## Lipid characteristics
"""

# ╔═╡ 0d352c11-05fe-4980-b75c-84ba8f9311d5
md"""
Each lipid ID corresponds to the nomenclature of a molecule that has been profiled in our lipidomic:
"""

# ╔═╡ b89c5eb8-6b2b-49be-a0f7-cf0c7ea96e92

# lipRefTable = dfLipidsXref[sample(1:size(dfLipidsXref)[1],3, replace=false, ordered=true), :]
# pretty_table(lipRefTable, names(lipRefTable); backend = :html, tf = mytf)
dfLipidsXref[sample(1:size(dfLipidsXref)[1],3, replace=false, ordered=true), :]

# ╔═╡ 01721454-c561-4a7c-816d-e310ae24f4ca
md"""
Frequency table of the present lipids main class according to the Grammar of Succinct Lipid Nomenclature (Goslin): 
"""

# ╔═╡ aa046894-c027-44be-8088-3a01d25162f5
begin
########################
# Load Z raw all valid #
########################

# Get Zmat 
dfZrawAll =  DataFrame(CSV.File("../../data/data_processed/ZmatRawAll.csv"))

classLipids = freqtable(dfZrawAll.Class)
# DataFrame(Class = names(classLipids, 1), Count = collect(classLipids[:]))
# lipTable = DataFrame(Class = names(classLipids, 1), Count = collect(classLipids[:]))
# pretty_table(lipTable, names(lipTable); backend = :html, tf = mytf)
	DataFrame(Class = names(classLipids, 1), Count = collect(classLipids[:]))
end

# ╔═╡ dee5f78c-231e-4c84-8bbc-88fa21864fb9
md"""
Since triglycerides (TG) are one of the important constituents of the lipid fraction of the human body, and represent the main lipid component of our dietary fat and fat depots, we are interested to study variation in triglyceride composition.
"""

# ╔═╡ 928fd773-8895-4ab7-8fcc-d497eeaa2f39
md"""
## Using matrix linear models to study variation in triglyceride composition

Matrix linear models provide a framework for studying associations in high-throughput data using bilinear models.

$$Y= X B Z^\prime + E,$$

where
- ``Y`` is the outcome data matrix (triglyceride levels)
- ``X`` is a design matrix constructed from individual covariates (phenotype status, whether they were taking fish oil)
- ``Z`` is a design matrix constructed from outcome covariates (information about the triglycerides)
- ``E`` is random error assumed to have mean zero, independent across individuals, but possibly correlated across outcomes

$$V(vec(E)) = \Sigma \otimes I.$$
"""

# ╔═╡ 82a16ada-6ec4-412d-99a5-72832d3b2212
md"""
 $(load("../../images/matrixlinearmodel.png"))
(Liang, J. W. and Sen, S., "Sparse matrix linear models for structured high-throughput data",2017, eprint arXiv:1712.05767)
"""

# ╔═╡ b7f70eff-dd78-4513-9870-d27341fe68a5
md"""
## Data analysis
"""

# ╔═╡ 8d10cf99-7bbe-4270-bb75-d35bc4298c81
begin
	# Select sub-class of lipids:
rdZ = ["Triglycerides", "Phospholipids"][1];
	
	# Select the X design matrix model:
if hasFishOil
    selectXdesign = false   
    rdX = ifelse(selectXdesign, 
            ["true" => "CN - No Fish Oil, CN - Fish Oil, CS - No Fish Oil, CS - Fish Oil (CN: Control, CS: Statin Intolerant)"][1][1], 
            ["false"=>"Intercept, Group, Fish Oil, Interaction"][1][1])
           
else
    rdX = ["false"=>"Intercept, Group"][1][1]
end;
	
	
# Get Zmat 
	# Get Zmat 
fZrawAll = "../../data/data_processed/ZmatRawAll.csv"
fZrawTG = "../../data/data_processed/ZmatRawTG.csv"
fZrawP = "../../data/data_processed/ZmatRawPhos.csv"
fZrawPC = "../../data/data_processed/ZmatRawPhosPC.csv"
	# fZrawPE = "../../data/data_processed/ZmatRawPhosPE.csv"

fZraw = Dict("Triglycerides"=> fZrawTG,
             "Phospholipids"=> fZrawP,
             "Phospholipids: PC"=> fZrawPC);

dfZraw =  DataFrame(CSV.File(fZraw[rdZ]))
m =  length(dfZraw.Lipids); #ifelse(pairingFlag, size(df)[2]-5, size(df)[2]-2)

is4ways = eval(Meta.parse(rdX));

# ZI is identity matrix 
ZI = convert(Array{Float64, }, collect(I(m)));
CoefZI, CIZI, TstatZI = getCoefs(df, ZI, responseSelection = dfZraw.lipID,
                                 isunpaired = pairingFlag, 
                                 hasFishOil = hasFishOil, 
                                 is4ways = is4ways);


# Get X and Y 
if hasFishOil
    if is4ways
        X, Y = getXY4ways(df, responseSelection = dfZraw.lipID,
                 isunpaired = pairingFlag);
        covarSelection = ["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil",
                      "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]
    else
        X, Y = getXY(df, responseSelection = dfZraw.lipID,
                 isunpaired = pairingFlag);
        covarSelection = ["1"=> "Intercept", "2"=> "Group",
                      "3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]
    end
else
    X, Y = getXYnoFishOil(df, responseSelection = dfZraw.lipID,
                 isunpaired = pairingFlag);
    covarSelection = ["1"=> "Intercept", "2"=> "Group"]
end

# nameX contains labels for plotting
if pairingFlag
    if is4ways
        namesX = ["CN-No Fish Oil" "CN-Fish Oil" "CS-No Fish Oil" "CS-Fish Oil"]
    else
        namesX = ["Intercept" "Group" "Fish Oil" "Interaction Group-Fish Oil"]
    end
else
    namesX = ["Intercept" "Interaction Group-Fish Oil"]
end;
end;

# ╔═╡ c49c47ac-b4c7-4368-9669-3c8c66c81126
md"""
### X matrix
"""

# ╔═╡ e7e9a683-7515-4c50-8dda-c24facd0ec1d
X[1:7,:]

# ╔═╡ 265da14c-8000-4f74-9979-69b069d038d7
md"""
Two-way contingency tables considering `Group` and `FishOil` variables from the unpaired data: 
"""

# ╔═╡ 01d810dc-c8f8-48b1-9cfa-a1476b2df3db
begin
xTable = freqtable(df[:, [:FishOil, :Group]], :FishOil, :Group)
matXfreq = collect(xTable[:,:]);
matXfreq = vcat(matXfreq, sum(matXfreq, dims= 1))
matXfreq = hcat(matXfreq, sum(matXfreq, dims= 2))
dfFreqX = DataFrame(matXfreq, :auto)
rename!(dfFreqX, Symbol.(vcat(names(xTable,2), "Total")))
insertcols!(dfFreqX, 1, :_ => vcat("Fish Oil: ".*(names(xTable,1)), Symbol("Total")))
end

# ╔═╡ 222bfccb-2c58-4e59-94ca-5ce490f63419
md"""
### Y matrix
"""

# ╔═╡ 7784f26b-ca1d-496a-8db7-4a2bfd6cdbae
Y[1:7,1:7]

# ╔═╡ 150d2d75-2099-48dc-8603-37bd8be2b0ae
md"""
### Z matrix

Triglycerides'useful information to build our Z matrix:
"""

# ╔═╡ 34aaa44a-15af-47d7-af12-5ede12f75255
dfZraw[1:7,3:end]

# ╔═╡ 113840b3-dce7-4f0c-bd93-0e1bb76c6a4e
md"""
## Lipid characteristic distributions
"""

# ╔═╡ 10adcba6-ce5d-4a5f-8740-789ccb9c6825
begin
oxiTable = freqtable(dfZraw.Oxidation)
DataFrame(Oxidized = names(oxiTable, 1), Count = collect(oxiTable[:]))
end

# ╔═╡ f6141719-2ff5-45af-8878-7fc681da5679
begin
plasmalogenTable = freqtable(dfZraw.Plasmalogen)
DataFrame(Plasmalogen = names(plasmalogenTable, 1), Count = collect(plasmalogenTable[:]))
end

# ╔═╡ 87151526-4525-49e2-908b-780caa813642
histogram2d(dfZraw.Total_DB, dfZraw.Total_C,
            c= :matter, aspectratio = 0.35, xlim = [0,15],
            colorbar_title = "Number of Triglycerides",
            xlabel= "Total Double Bonds", ylabel =  "Total Carbons",
            title = "Number of triglycerides by the number of carbon \n atoms and the number of unsaturated bonds \n")


# ╔═╡ 651633a7-6172-4bee-8e3c-d53533c9735f
md"""
### Scatterplots of lipid effects analyzed individually
"""

# ╔═╡ 29b21dfe-8d2c-4572-b0ee-2836a74b80ea
idxColXmat = 3;
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#

# ╔═╡ 56a95af0-87ae-4b64-b6ce-45f1af84b0f7
begin
a = 0.25
zlim = maximum(abs.(CoefZI[idxColXmat,:]))
scatter(dfZraw.Total_DB .+ rand(Uniform(-a,a),length(dfZraw.Total_DB)),
    dfZraw.Total_C,
    zcolor = TstatZI[idxColXmat,:], colorbar = true, m = (:bluesreds, 0.8), 
    colorbar_title = string(" \nEffect Size of ",namesX[idxColXmat]), xticks=0:1:14, 
    clims = (-zlim, zlim), xlabel= "Total Double Bonds", ylabel =  "Total Carbons",
    legend = false, grid = false, tickfontsize=8, 
	right_margin = 10mm,  format = :svg,
    title = string("Total Carbons according to Total Double Bonds \n with colorbar based on effect size of \n", namesX[idxColXmat]) )
savefig(string("../../images/scatterCoef_",namesX[idxColXmat],".svg"))
end

# ╔═╡ 3a746365-165a-43d9-aaa0-4111778172b5
begin
zlim2 = maximum(abs.(TstatZI[idxColXmat,:]))
scatter(dfZraw.Total_DB .+ rand(Uniform(-a,a),length(dfZraw.Total_DB)),
    dfZraw.Total_C,
    zcolor = TstatZI[idxColXmat,:], colorbar = true, m = (:bluesreds, 0.8), 
    colorbar_title = string("\nT-statistics of ",namesX[idxColXmat]), xticks=0:1:14,
    clims = (-zlim2, zlim2), #yaxis= false, xaxis = false,
    xlabel= "Number of Double Bonds", ylabel =  "Carbon Chain Length",
    legend = false, grid = false, tickfontsize=8, 
	right_margin = 10mm,  format = :svg,
    title = string("Total Carbons according to Total Double Bonds \n with colorbar based on T-statistics of\n ", namesX[idxColXmat]) 
)
# savefig(string("../../images/scatterTstats_",namesX[idxRow],".svg"))
end

# ╔═╡ 02b1e948-48f0-4451-917c-24fa62e8d465
md"
This scatter plot indicates a strong association between fish oil and the levels of polyunsaturated triglycerides.
"

# ╔═╡ ffab514f-bb9a-472d-8bed-f68259865b9e
md"""
## Modeling Decisions
"""

# ╔═╡ 4a6ec997-fc98-4ed3-84b8-7b88bd040cc8
begin
########################
# Z matrix collections #
########################

dfZlinear = DataFrame(Total_C = dfZraw.Total_C,
            Total_DB = dfZraw.Total_DB,
            Interaction = dfZraw.Total_C.*dfZraw.Total_DB);

dfZtcdb = DataFrame(intercept = ones(Float64, size(dfZraw)[1]),
            # TotalC_0_45 = dfZraw.Total_C .< 40,
            TotalC_45_55 = ((dfZraw.Total_C .>= 45) .& (dfZraw.Total_C .< 55))*1,
            TotalC_55_65 = ((dfZraw.Total_C .>= 55) .& (dfZraw.Total_C .< 65))*1,
            TotalC_65 = (dfZraw.Total_C .>= 65)*1,
            # TotalDB_0_3 = dfZraw.Total_DB .< 3,
            TotalDB_3_6 = ((dfZraw.Total_DB .>= 3) .& (dfZraw.Total_DB .< 6))*1,
            TotalDB_6_9 = ((dfZraw.Total_DB .>= 6) .& (dfZraw.Total_DB .< 9))*1,
            TotalDB_9 = (dfZraw.Total_DB .>= 9)*1	);

dfZtc = DataFrame(intercept = ones(Float64, size(dfZraw)[1]),
            # TotalC_0_45 = dfZraw.Total_C .< 40,
            TotalC_40_45 = ((dfZraw.Total_C .>= 40) .& (dfZraw.Total_C .< 45))*1,
            TotalC_45_50 = ((dfZraw.Total_C .>= 45) .& (dfZraw.Total_C .< 50))*1,
            TotalC_50_55 = ((dfZraw.Total_C .>= 50) .& (dfZraw.Total_C .< 55))*1,
            TotalC_55_60 = ((dfZraw.Total_C .>= 55) .& (dfZraw.Total_C .< 60))*1,
            TotalC_60_65 = ((dfZraw.Total_C .>= 60) .& (dfZraw.Total_C .< 65))*1,
            TotalC_65_70 = ((dfZraw.Total_C .>= 65) .& (dfZraw.Total_C .< 70))*1,
            TotalC_70 = (dfZraw.Total_C .>= 70)*1  );

dfZdbcat = DataFrame(intercept = ones(Float64, size(dfZraw)[1]),
            DoubleBound_1 = (dfZraw.Total_DB .== 1)*1,
            DoubleBound_2 = (dfZraw.Total_DB .== 2)*1,
            DoubleBound_3 = (dfZraw.Total_DB .== 3)*1,
            DoubleBound_4 = (dfZraw.Total_DB .== 4)*1,
            DoubleBound_5 = (dfZraw.Total_DB .== 5)*1,
            DoubleBound_6 = (dfZraw.Total_DB .== 6)*1,
            DoubleBound_7 = (dfZraw.Total_DB .== 7)*1,
            DoubleBound_8 = (dfZraw.Total_DB .== 8)*1,
            DoubleBound_9 = (dfZraw.Total_DB .== 9)*1,
            MoreThan9 = (dfZraw.Total_DB .> 9)*1 );

dfZoxy = DataFrame(intercept = ones(Float64, size(dfZraw)[1]),
            Oxidized = (dfZraw.Oxidation .== "yes").*1.0);

dfZplsm = DataFrame(intercept = ones(Float64, size(dfZraw)[1]),
            plasmenyl = (dfZraw.Plasmalogen .== "plasmenyl")*1,
            plasmanyl = (dfZraw.Plasmalogen .== "plasmanyl")*1 );

dfZphsph = DataFrame(PC = ones(Float64, size(dfZraw)[1]),
            PA = (occursin.(r"PA", dfZraw.Lipids))*1.0,
            PE = (occursin.(r"PE", dfZraw.Lipids))*1.0 );


#########################
# Z matrix dictionaries # 
#########################

dfZ = Dict("1"=>dfZlinear, #"~ Total C*Total DB"
            "2"=>dfZtcdb, # "factor(Total C, Total DB)"
            "3"=>dfZtc, # "factor(Total C)"
            "4"=>dfZdbcat, # "factor(Total DB)"
            "5"=>dfZoxy, # "factor(Oxidation)"
            "6"=>dfZplsm, # "factor(Plasmalogen)"
            "7"=>dfZphsph); # "factor(PC, PA, PE)"

classZ = Dict("1"=>"Chain length and degree of unsaturation", 
            "2"=>"Carbon Chain Length and Degree of Unsaturation",
            "3"=>"Carbon Chain Length",
            "4"=>"Degree of Unsaturation",
            "5"=>"Oxidation",
            "6"=>"Plasmalogen",
            "7"=>"Phospholipids class");
end;

# ╔═╡ 3ffba375-a00b-4084-89d3-a5a2275631ba
md"""
### 1. Z matrix: triglycerides classified by the number of carbon chain length 
"""

# ╔═╡ 2a582157-83a3-437c-b7cd-b3996c3515cb
begin
if rdZ == "Triglycerides"
    slct2Tc = "3"
    #=  ["1"=> "Continuous: Total C, Total DB, Total C:Total DB", 
         "2"=> "Categorical: Total C, Total DB",
         "3"=> "Categorical: Total C", 
         "4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "6"=> "Categorical: Plasmalogen"] =#
else
    slct2Tc = "4"
    #=  ["4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "7"=> "Categorical: Phospholipid(PC, PA, PE)"] =#
end;

namesZTc = names(dfZ[slct2Tc]);
ZTc = Matrix{Float64}(dfZ[slct2Tc]);
CoefZTc, CIZTc, TstatZTc = getCoefs(df, ZTc, responseSelection = dfZraw.lipID,
                              isunpaired = pairingFlag, hasFishOil = hasFishOil, is4ways = is4ways);
	dfZ[slct2Tc][1:5,:]
end

# ╔═╡ 632a4db4-971d-4292-a361-c99669533c77
md"""
#### Effect Size of Matrix Linear Model regression coefficients
"""

# ╔═╡ b8fd0e80-dd7c-4e52-9773-cd8cd51d1038
plot(transpose(CoefZTc[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Tc], ylabel = "Effect size",
    legend =:topleft,
	lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZTc)[2]), namesZTc),xrotation = 45, size = (700,550),
	bottom_margin = 10mm, format = :svg,
	foreground_color_legend = nothing,
    title = string("Matrix Linear Model Coefficients Estimates") )
# savefig(string("../../images/mlmCoef_","Saturation",".svg"))

# ╔═╡ 6a499da5-1896-4856-9c31-e073db0a7398
md"""
#### T-statistics
"""

# ╔═╡ 0a1d0171-4ca8-4c35-aab7-2af9085d13fe
plot(transpose(TstatZTc[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Tc], ylabel = "T-statistics",
    legend =:topleft, foreground_color_legend = nothing,
	lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZTc)[2]), namesZTc),xrotation = 45, size = (700,550),
	bottom_margin = 10mm, format = :svg,
    # title = string("Matrix Linear Model T-statistics of \nCoefficients Estimates")
)
# savefig(string("../../images/mlmTstats_","Saturation",".svg"))

# ╔═╡ 59945ce5-44df-4af1-a9b1-1ed9b9516d38
md"""
In this model, we consider only the scale of carbon chain length in the Z matrix. The fish oil effect seems to significantly increase the level of triglycerides having a carbon chain length of more than 50 and less than 70. The interaction variable reflects the difference between the control and the symptomatic group. Therefore, we observe that the fish oil effect is stronger among the symptomatic group than the control group for the triglycerides with carbon chain length ranging between 55 and 60. 
"""

# ╔═╡ 0b5ab124-7232-4ce0-bac0-a9511f102548
md"""
#### Confidence intervals of coefficients
"""

# ╔═╡ f6eb557a-b2ed-498d-b854-bbdc784449de
begin
	idxColXmatCITc = 3
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#
	
	
plot((CoefZTc[idxColXmatCITc,:]), ribbon = CIZTc[idxColXmatCITc,:],
        label= namesX[idxColXmatCITc],xlabel= classZ[slct2Tc], ylabel = "Effect size",
        legend =:topleft, foreground_color_legend = nothing,
		lw = 2, marker = :circle,
        xticks = (collect(1:size(ZTc)[2]), namesZTc), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCITc]," according to ", classZ[slct2Tc]) 
)
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxRow],"_Saturation",".svg"))
end

# ╔═╡ 6c973af4-aeab-4bc3-8936-a239d01d00f6
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZTcDb,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCITc = 3
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZTc, 
		Estimate = round.(CoefZTc[idxColXmatCITc,:]; digits = 3),
		SE = round.(CIZTc[idxColXmatCITc,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZTc[idxColXmatCITc,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZTc[idxColXmatCITc,:])).*2; digits = 3))
end

# ╔═╡ d9f8445a-f9b1-49b9-8342-3cca5735f152
let
	idxColXmatCITc = 4
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#
	
	
plot((CoefZTc[idxColXmatCITc,:]), ribbon = CIZTc[idxColXmatCITc,:],
        label= namesX[idxColXmatCITc],xlabel= classZ[slct2Tc], ylabel = "Effect size",
        legend =:topleft, foreground_color_legend = nothing,
		lw = 2, marker = :circle,
        xticks = (collect(1:size(ZTc)[2]), namesZTc), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm,  format = :SVG, 
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCITc]," \naccording to ", classZ[slct2Tc])
)
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxRow],"_Saturation",".svg"))
end

# ╔═╡ 7295b3e8-e1d9-4c5e-879e-413198fb34d8
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZTcDb,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCITc = 4
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZTc, 
		Estimate = round.(CoefZTc[idxColXmatCITc,:]; digits = 3),
		SE = round.(CIZTc[idxColXmatCITc,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZTc[idxColXmatCITc,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZTc[idxColXmatCITc,:])).*2; digits = 3))
end

# ╔═╡ ec8c9a5a-99ee-4b15-8a66-a23e114d08de
md"""
### 2. Z matrix: triglycerides unsaturation classified by the number of double bonds 
"""

# ╔═╡ b37881f4-acdb-494a-aec1-db484c1ff9b4
begin
if rdZ == "Triglycerides"
    slct2Db = "4"
    #=  ["1"=> "Continuous: Total C, Total DB, Total C:Total DB", 
         "2"=> "Categorical: Total C, Total DB",
         "3"=> "Categorical: Total C", 
         "4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "6"=> "Categorical: Plasmalogen"] =#
else
    slct2Db = "4"
    #=  ["4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "7"=> "Categorical: Phospholipid(PC, PA, PE)"] =#
end;

namesZDb = names(dfZ[slct2Db]);
ZDb = Matrix{Float64}(dfZ[slct2Db]);
CoefZDb, CIZDb, TstatZDb = getCoefs(df, ZDb, responseSelection = dfZraw.lipID,
                              isunpaired = pairingFlag, hasFishOil = hasFishOil, is4ways = is4ways);
	dfZ[slct2Db][1:5,:]
end

# ╔═╡ 7b28028b-fc27-4297-abed-44d3afea9fdb
md"""
#### Effect Size of Matrix Linear Model regression coefficients
"""

# ╔═╡ 9bfa2c3a-aa51-4eec-af3b-b4960e86db13
plot(transpose(CoefZDb[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Db], ylabel = "Effect size",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZDb)[2]), namesZDb),xrotation = 45, size = (800,550),  
	bottom_margin = 10mm, format = :svg,
    title = string("Matrix Linear Model Coefficients Estimates") )
# savefig(string("../../images/mlmCoef_","Saturation",".svg"))

# ╔═╡ 2a5d6ba9-1f78-4385-89ea-34ac810dda09
md"""
#### T-statistics
"""

# ╔═╡ cda7e0c8-58d1-4cb6-8114-d4c22994b62d
begin
plot(transpose(TstatZDb[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Db], ylabel = "T-statistics",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZDb)[2]), namesZDb),xrotation = 45, size = (800,550), 
	bottom_margin = 10mm, format = :svg,
    title = string("Matrix Linear Model T-statistics of \nCoefficients Estimates") )
# savefig(string("../../images/mlmTstats_","Saturation",".svg"))
end

# ╔═╡ 3f6e13eb-7edd-4f56-a254-358a85fb4cd0
md"""
From the two figures above, we can observe that the fish oil effect substantially increases the levels of polyunsaturated triglycerides with six or more double bonds in both groups, control and symptomatic. By looking at the interaction group-fish oil covariate, we can say that the fish oil effect on the polyunsaturated triglycerides ranging between 7 and 9 double bonds is stronger in the symptomatic group than in control one.
"""

# ╔═╡ 6daec259-426d-480d-87c8-496440c32187
md"""
#### Confidence intervals of coefficients
"""

# ╔═╡ 540d482a-582c-46aa-9319-ac29adcfbd5a
begin
	idxColXmatCIDb = 3
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#
	
	
plot((CoefZDb[idxColXmatCIDb,:]), ribbon = CIZDb[idxColXmatCIDb,:],
        label= namesX[idxColXmatCIDb],xlabel= classZ[slct2Db], ylabel = "Effect size",
        legend =:outerright, lw = 2, marker = :circle,
        xticks = (collect(1:size(ZDb)[2]), namesZDb), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCIDb]," \naccording to ", classZ[slct2Db]) )
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxRow],"_Saturation",".svg"))
end

# ╔═╡ cb14e0fb-c864-4a91-bf9d-d97b08177837
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZTcDb,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCIDb = 3
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZDb, 
		Estimate = round.(CoefZDb[idxColXmatCIDb,:]; digits = 3),
		SE = round.(CIZDb[idxColXmatCIDb,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZDb[idxColXmatCIDb,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZDb[idxColXmatCIDb,:])).*2; digits = 3))
end

# ╔═╡ a6d6da76-c2b2-43f5-a62d-89714db7af70
let
	idxColXmatCIDb = 4
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#
	
	
plot((CoefZDb[idxColXmatCIDb,:]), ribbon = CIZDb[idxColXmatCIDb,:],
        label= namesX[idxColXmatCIDb],xlabel= classZ[slct2Db], ylabel = "Effect size",
        legend =:outerright, lw = 2, marker = :circle,
        xticks = (collect(1:size(ZDb)[2]), namesZDb), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCIDb]," \naccording to ", classZ[slct2Db],"\n") )
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxRow],"_Saturation",".svg"))
end

# ╔═╡ a025ef1a-e96c-492f-9eb1-4347bce83606
md"""
###  3. Z matrix: triglycerides grouped by total number of carbon adjusted to the number of double bonds 
"""

# ╔═╡ 48dee03b-9ceb-48fb-a002-28410f3fe1df
begin
if rdZ == "Triglycerides"
    slct2TcDb = "2"
    #=  ["1"=> "Continuous: Total C, Total DB, Total C:Total DB", 
         "2"=> "Categorical: Total C, Total DB",
         "3"=> "Categorical: Total C", 
         "4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "6"=> "Categorical: Plasmalogen"] =#
else
    slct2TcDb = "4"
    #=  ["4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "7"=> "Categorical: Phospholipid(PC, PA, PE)"] =#
end;

namesZTcDb = names(dfZ[slct2TcDb]);
ZTcDb = Matrix{Float64}(dfZ[slct2TcDb]);
CoefZTcDb, CIZTcDb, TstatZTcDb = getCoefs(df, ZTcDb, responseSelection = dfZraw.lipID,
                              isunpaired = pairingFlag, hasFishOil = hasFishOil, is4ways = is4ways);
	dfZ[slct2TcDb][1:5,:]
end

# ╔═╡ 8e76d171-a6c9-4009-b1b3-de13ea84c758
md"""
#### Effect Size of Matrix Linear Model regression coefficients
"""

# ╔═╡ bf2da2d2-d11e-4c6a-9a2b-60e6ca41a437
begin
plot(transpose(CoefZTcDb[:,1:end]), 
    label= namesX, xlabel= classZ[slct2TcDb], ylabel = "Effect size",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
	bottom_margin = 10mm, format = :svg,
    xticks = (collect(1:size(ZTcDb)[2]), namesZTcDb),xrotation = 45, size = (800,550),  
    title = string("Matrix Linear Model Coefficients Estimates") )
# savefig(string("../../images/mlmCoef_","Saturation",".svg"))
end

# ╔═╡ 37a3f827-e270-4776-8edc-179130a59cd4
md"""
#### T-statistics
"""

# ╔═╡ 610f8dfc-a888-44a8-b292-a250dcedaa7b
begin
plot(transpose(TstatZTcDb[:,1:end]), 
    label= namesX, xlabel= classZ[slct2TcDb], ylabel = "T-statistics",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
	bottom_margin = 10mm, format = :svg,
    xticks = (collect(1:size(ZTcDb)[2]), namesZTcDb),xrotation = 45, size = (800,550),  
    title = string("Matrix Linear Model T-statistics of \nCoefficients Estimates") )
# savefig(string("../../images/mlmTstats_","Saturation",".svg"))
end

# ╔═╡ 106abb2b-8b79-4d18-a436-df411160179c
md"""
The figure above relay more clearly the same information observed in the previous scatter plot (Total Carbons according to Total Double Bonds with colorbar based on T-statistics of Fish Oil). Fish oil substantially affects the levels of polyunsaturated triglycerides with six or more double bonds on both groups. We can see that the effect of carbon chain length is not significant anymore when it is adjusted to the number of double bonds.
"""

# ╔═╡ 32c59be3-efc8-4854-a2c5-aa928accfc9f
md"""
#### Confidence intervals of coefficients
"""

# ╔═╡ d1f7b362-0eba-4c4d-9c89-7546f6c9a1f6
begin
	idxColXmatCITcDb = 3
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#
	
plot((CoefZTcDb[idxColXmatCITcDb,:]), ribbon = CIZTcDb[idxColXmatCITcDb,:],
        label= namesX[idxColXmatCITcDb],xlabel= classZ[slct2TcDb], ylabel = "Effect size",
    legend =:topleft, foreground_color_legend = nothing,
	lw = 2, marker = :circle,
        xticks = (collect(1:size(ZTcDb)[2]), namesZTcDb), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCITcDb]," \naccording to ", classZ[slct2TcDb],"\n") 
)
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxColXmatCITcDb],"_Saturation",".svg"))
end

# ╔═╡ 7799abbe-5d61-49e1-8e06-8f56565f87de
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZTcDb,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCIDb = 4
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZDb, 
		Estimate = round.(CoefZDb[idxColXmatCIDb,:]; digits = 3),
		SE = round.(CIZDb[idxColXmatCITcDb,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZDb[idxColXmatCIDb,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZDb[idxColXmatCIDb,:])).*2; digits = 3))
end

# ╔═╡ b0211ec8-ce1c-45aa-b346-e0d0df0b471a
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZTcDb,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCITcDb = 3
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZTcDb, 
		Estimate = round.(CoefZTcDb[idxColXmatCITcDb,:]; digits = 3),
		SE = round.(CIZTcDb[idxColXmatCITcDb,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZTcDb[idxColXmatCITcDb,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZTcDb[idxColXmatCITcDb,:])).*2; digits = 3))
end

# ╔═╡ 8936338a-5ab6-4186-bc9a-0fe91eb384bc
let
	idxColXmatCITcDb = 4
#=
Fish Oil - selectXdesign false
["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]

Fish Oil - selectXdesign true
["1"=> "CN-No Fish Oil (intercept)", "2"=> "CN-Fish Oil", "3"=> "CS-No Fish Oil", "4"=> "CS-Fish Oil"]

No Fish Oil
["1"=> "Intercept", "2"=> "Group"]
=#
	
plot((CoefZTcDb[idxColXmatCITcDb,:]), ribbon = CIZTcDb[idxColXmatCITcDb,:],
        label= namesX[idxColXmatCITcDb],xlabel= classZ[slct2TcDb], ylabel = "Effect size",
    legend =:topleft, foreground_color_legend = nothing,
	lw = 2, marker = :circle,
        xticks = (collect(1:size(ZTcDb)[2]), namesZTcDb), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCITcDb]," \naccording to ", classZ[slct2TcDb],"\n") 
)
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxColXmatCITcDb],"_Saturation",".svg"))
end

# ╔═╡ aee948c5-28e2-4cb4-b228-bfe3541fbd8d
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZTcDb,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCITcDb = 4
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZTcDb, 
		Estimate = round.(CoefZTcDb[idxColXmatCITcDb,:]; digits = 3),
		SE = round.(CIZTcDb[idxColXmatCITcDb,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZTcDb[idxColXmatCITcDb,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZTcDb[idxColXmatCITcDb,:])).*2; digits = 3))
end

# ╔═╡ 4b3ceee6-0753-427d-a859-6b2570dc24fb
md"""
###  4. Z matrix: triglycerides oxidation
"""

# ╔═╡ 2ae650a3-0a84-4a83-8614-eb8f4b72d654
begin
if rdZ == "Triglycerides"
    slct2Ox = "5"
    #=  ["1"=> "Continuous: Total C, Total DB, Total C:Total DB", 
         "2"=> "Categorical: Total C, Total DB",
         "3"=> "Categorical: Total C", 
         "4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "6"=> "Categorical: Plasmalogen"] =#
else
    slct2Ox = "4"
    #=  ["4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "7"=> "Categorical: Phospholipid(PC, PA, PE)"] =#
end;

namesZOx = names(dfZ[slct2Ox]);
ZOx = Matrix{Float64}(dfZ[slct2Ox]);
CoefZOx, CIZOx, TstatZOx = getCoefs(df, ZOx, responseSelection = dfZraw.lipID,
                              isunpaired = pairingFlag, hasFishOil = hasFishOil, is4ways = is4ways);
end;

# ╔═╡ bc5a41e4-96ad-4c06-8afb-0e4d4f61f1ec
md"""
#### Effect Size of Matrix Linear Model regression coefficients
"""

# ╔═╡ 04642132-7128-4c8c-ae4b-11fb4a9419ff
plot(transpose(CoefZOx[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Ox], ylabel = "Effect size",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
	bottom_margin = 10mm, format = :svg,
    xticks = (collect(1:size(ZOx)[2]), namesZOx),xrotation = 45, size = (800,550),  
    title = string("Matrix Linear Model Coefficients Estimates") )
# savefig(string("../../images/mlmCoef_","Saturation",".svg"))

# ╔═╡ c9a2efdf-8aa8-4270-b707-97f8c4e9cca4
md"""
#### T-statistics
"""

# ╔═╡ 62edb8f3-8e2c-43d7-9b66-a45f526bf201
plot(transpose(TstatZOx[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Ox], ylabel = "T-statistics",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZOx)[2]), namesZOx),xrotation = 45, size = (800,550),
	bottom_margin = 10mm, format = :svg,
    title = string("Matrix Linear Model T-statistics of \nCoefficients Estimates") )
# savefig(string("../../images/mlmTstats_","Saturation",".svg"))

# ╔═╡ 30b4e409-5ea9-4109-bd74-bc9221fc721a
md"""
From the two preceding figures, we can see that the fish oil effect substantially decreases the levels of oxidized triglycerides similarly in both control and symptomatic groups. 
"""

# ╔═╡ 5be338dd-afc9-4e15-b308-53d424450217
md"""
#### Confidence intervals of coefficients
"""

# ╔═╡ f3866823-b2fa-4575-8304-23f4c4e16f96
begin
idxColXmatCIOx = 3
# ["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]
plot((CoefZOx[idxColXmatCIOx,:]), ribbon = CIZOx[idxColXmatCIOx,:],
        label= namesX[idxColXmatCIOx],xlabel= classZ[slct2Ox], ylabel = "Effect size",
        legend =:outerright, lw = 2, marker = :circle,
        xticks = (collect(1:size(ZOx)[2]), namesZOx), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCIOx]," according to ", classZ[slct2Ox],"\n") )
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxRow],"_Saturation",".svg"))
end

# ╔═╡ 12a38212-70db-4155-960d-e989b43e750d
let
	# To estimate P-values based on the T-statistics  
	# tStats, pVals = getPermPvals(df, ZOx,
	# 	nPerms = 5000,
	# 	responseSelection = dfZraw.lipID,
	# 	isunpaired = pairingFlag,
	# 	hasFishOil = hasFishOil, is4ways = is4ways);
	idxColXmatCIOx = 3
	sampleN = size(X)[1]
	DataFrame( Coefficients = namesZOx, 
		Estimate = round.(CoefZOx[idxColXmatCIOx,:]; digits = 3),
		SE = round.(CIZOx[idxColXmatCIOx,:]./(1.96); digits = 4),
		Tstatistics = round.(TstatZOx[idxColXmatCIOx,:]; digits = 3), 
		Pvalues = round.(ccdf.(TDist(sampleN-1),
			abs.(TstatZOx[idxColXmatCIOx,:])).*2; digits = 3))
	end

# ╔═╡ 7087e085-978a-487e-ba1a-4fc354fd1e5d
md"
List of the oxidized triglycerides:
"

# ╔═╡ 8bdbf6f0-0deb-4bf3-ab1a-c6edf1af9f44
filter(row -> row.Oxidation == "yes", dfZraw)[:,[1,3]]

# ╔═╡ 261b196f-fe8f-4e7a-a37d-443ddd11346a
md"""
### 5. Z matrix: triglycerides plasmalogen classes
"""

# ╔═╡ 6b1c9a67-4365-450b-af34-bd4df85b82b8
begin
if rdZ == "Triglycerides"
    slct2Pl = "6"
    #=  ["1"=> "Continuous: Total C, Total DB, Total C:Total DB", 
         "2"=> "Categorical: Total C, Total DB",
         "3"=> "Categorical: Total C", 
         "4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "6"=> "Categorical: Plasmalogen"] =#
else
    slct2Pl = "4"
    #=  ["4"=> "Categorical: Total DB", 
         "5"=> "Categorical: Oxidation", 
         "7"=> "Categorical: Phospholipid(PC, PA, PE)"] =#
end;

namesZPl = names(dfZ[slct2Pl]);
ZPl = Matrix{Float64}(dfZ[slct2Pl]);
CoefZPl, CIZPl, TstatZPl = getCoefs(df, ZPl, responseSelection = dfZraw.lipID,
                              isunpaired = pairingFlag, hasFishOil = hasFishOil, is4ways = is4ways);
end;

# ╔═╡ 9c67d367-657e-4d0f-9745-11550f75af01
md"""
#### Effect Size of Matrix Linear Model regression coefficients
"""

# ╔═╡ 89159376-a626-4a09-960f-8054f36201ee
plot(transpose(CoefZPl[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Pl], ylabel = "Effect size",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZPl)[2]), namesZPl),xrotation = 45, size = (800,550),  
	bottom_margin = 10mm, format = :svg,
    title = string("Matrix Linear Model Coefficients Estimates") )
# savefig(string("../../images/mlmCoef_","Saturation",".svg"))

# ╔═╡ 82be8d9d-9275-4ace-ba90-79c848b452ba
md"""
#### T-statistics
"""

# ╔═╡ ee2027ca-fba7-4b11-a596-babba7387ebb
plot(transpose(TstatZPl[:,1:end]), 
    label= namesX, xlabel= classZ[slct2Pl], ylabel = "T-statistics",
    legend =:outerright, lw = 2, marker = :circle, grid = false,
    xticks = (collect(1:size(ZPl)[2]), namesZPl),xrotation = 45, size = (800,550),
	bottom_margin = 10mm, format = :svg,
    title = string("Matrix Linear Model T-statistics of \nCoefficients Estimates") )
# savefig(string("../../images/mlmTstats_","Saturation",".svg"))

# ╔═╡ b47aa87d-58d7-4000-a53e-105cfcbf6229
md"""
From the two figures above, it seems that the fish oil effect substantially decreases the levels of plasmenyl triglycerides similarly in both control and symptomatic groups.
"""

# ╔═╡ ec0e0778-b2bf-41bd-8342-40c24c948d9d
md"""
#### Confidence intervals of coefficients
"""

# ╔═╡ 2d06d398-a61a-4d92-80db-f2e3c857aec8
begin
idxColXmatCIPl = 3
# ["1"=> "Intercept", "2"=> "Group","3"=> "Fish Oil", "4"=> "Interaction Group-Fish Oil"]
plot((CoefZPl[idxColXmatCIPl,:]), ribbon = CIZPl[idxColXmatCIPl,:],
        label= namesX[idxColXmatCIPl],xlabel= classZ[slct2Pl], ylabel = "Effect size",
        legend =:outerright, lw = 2, marker = :circle,
        xticks = (collect(1:size(ZPl)[2]), namesZPl), xrotation = 45, size= (750, 500),
		bottom_margin = 10mm, format = :svg,
        title = string("Effect Size with its confidence of interval\n  of ", namesX[idxColXmatCIPl]," according to ", classZ[slct2Pl], "\n") )
hline!([0], color = :red, label = "")	
# savefig(string("../../images/mlmCI_",namesX[idxRow],"_Saturation",".svg"))
end

# ╔═╡ fbbad3be-c5b2-4719-aa1f-2dbe9182f0af
let
	# To estimate P-values based on the T-statistics  
	sampleN = 98
	ccdf(TDist(sampleN-1), abs.(TstatZPl[3,2])).*2
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataFramesMeta = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
FreqTables = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MatrixLM = "37290134-6146-11e9-0c71-a5c489be1f53"
Missings = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.9.11"
CategoricalArrays = "~0.10.2"
DataFrames = "~1.3.0"
DataFramesMeta = "~0.10.0"
Distributions = "~0.25.34"
FileIO = "~1.11.2"
FreqTables = "~0.4.5"
Images = "~0.25.0"
MatrixLM = "~0.1.3"
Missings = "~1.0.2"
MultivariateStats = "~0.8.0"
Plots = "~1.25.0"
PlutoUI = "~0.7.21"
PrettyTables = "~1.2.3"
StatsBase = "~0.33.13"
StatsModels = "~0.6.28"
StatsPlots = "~0.14.29"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "265b06e2b1f6a216e0e8f183d28e4d354eab3220"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "49f14b6c56a2da47608fe30aed711b5882264d7a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.11"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "c308f209870fdbd84cb20332b6dfaf14bf3387f8"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.2"

[[Chain]]
git-tree-sha1 = "cac464e71767e8a04ceee82a889ca56502795705"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.8"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "2e993336a3f68216be91eb8ee4625ebbaba19147"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.0"

[[DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "OrderedCollections", "Reexport"]
git-tree-sha1 = "ab4768d2cc6ab000cd0cec78e8e1ea6b03c7c3e2"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.10.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "7f3bec11f4bcd01bc1f507ebce5eadf1b0a78f47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.34"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "84f04fe68a3176a583b864e492578b9466d87f1e"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "3fe985505b4b667e1ae303c9ca64d181f09d5c05"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.3"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreqTables]]
deps = ["CategoricalArrays", "Missings", "NamedArrays", "Tables"]
git-tree-sha1 = "488ad2dab30fd2727ee65451f790c81ed454666d"
uuid = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
version = "0.4.5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78e2c69783c9753a91cdae88a8d432be85a2ab5e"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "92243c07e786ea3458532e199eb3feee0e7e08eb"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.4.1"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "7a20463713d239a19cbad3f6991e404aca876bda"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.15"

[[ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "a2951c93684551467265e0e32b577914f69532be"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.9"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f025b79883f361fa1bd80ad132773161d231fd9f"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+2"

[[ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "5581e18a74a5838bd919294a7138c2663d065238"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.0"

[[ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1d2d73b14198d10f7f12bf7f8481fd4b3ff5cd61"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.0"

[[ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "d0ac64c9bee0aed6fdbb2bc0e5dfa9a3a78e3acc"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.3"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "b4b161abc8252d68b13c5cc4a5f2ba711b61fec5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.3"

[[Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "35dc1cd115c57ad705c7db9f6ef5cc14412e8f00"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.0"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "ca99cac337f8e0561c6a6edeeae5bf6966a78d21"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.0"

[[IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "00019244715621f473d399e4e1842e479a69a42e"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "46b7834ec8165c541b0b5d1c8ba63ec940723ffb"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.15"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MatrixLM]]
deps = ["DataFrames", "Distributed", "GLM", "LinearAlgebra", "Random", "SharedArrays", "Statistics", "Test"]
git-tree-sha1 = "742d34bc491fa02269dd019a0fed8f7e40e549c0"
uuid = "37290134-6146-11e9-0c71-a5c489be1f53"
version = "0.1.3"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "6d105d40e30b635cfed9d52ec29cf456e27d38f8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.12"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "8789439a899b77f4fbb4d7298500a6a5781205bc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.0"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "b68904528fd538f1cb6a3fbc44d2abdc498f9e8e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.21"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[Quaternions]]
deps = ["DualNumbers", "LinearAlgebra"]
git-tree-sha1 = "adf644ef95a5e26c8774890a509a55b7791a139f"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "dbf5f991130238f10abbf4f2d255fb2837943c43"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.1.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "677488c295051568b0b79a77a8c44aa86e78b359"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.28"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "d6956cefe3766a8eb5caae9226118bb0ac61c8ac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.29"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "c342ae2abf4902d65a0b0bf59b28506a6e17078a"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.2"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─0b37e078-ce99-4268-9749-9933735129c3
# ╟─20574596-5e33-4be5-9958-e688fa712d84
# ╟─35fab8f4-a186-4d05-8369-91f574f9aa4c
# ╟─51459290-6b57-4ec7-8750-33e8014e7f42
# ╟─2b7def0b-703c-433a-857b-f83b8986f312
# ╟─536cf374-6611-4a71-8842-719e50300633
# ╟─a400ae63-60fe-4a98-b58a-f519edfc3644
# ╟─3b95f87e-6abe-4935-86ac-6598b5ee04bd
# ╟─2c7a26f6-cfba-48d9-a30a-22a0d1ef43ce
# ╟─67743f5d-454a-4be2-87fb-ac9e5062a059
# ╟─f5f01515-175f-4684-9758-512b709b4687
# ╟─0d352c11-05fe-4980-b75c-84ba8f9311d5
# ╟─b89c5eb8-6b2b-49be-a0f7-cf0c7ea96e92
# ╟─01721454-c561-4a7c-816d-e310ae24f4ca
# ╟─aa046894-c027-44be-8088-3a01d25162f5
# ╟─dee5f78c-231e-4c84-8bbc-88fa21864fb9
# ╟─928fd773-8895-4ab7-8fcc-d497eeaa2f39
# ╟─82a16ada-6ec4-412d-99a5-72832d3b2212
# ╟─b7f70eff-dd78-4513-9870-d27341fe68a5
# ╟─8d10cf99-7bbe-4270-bb75-d35bc4298c81
# ╟─c49c47ac-b4c7-4368-9669-3c8c66c81126
# ╟─e7e9a683-7515-4c50-8dda-c24facd0ec1d
# ╟─265da14c-8000-4f74-9979-69b069d038d7
# ╟─01d810dc-c8f8-48b1-9cfa-a1476b2df3db
# ╟─222bfccb-2c58-4e59-94ca-5ce490f63419
# ╟─7784f26b-ca1d-496a-8db7-4a2bfd6cdbae
# ╟─150d2d75-2099-48dc-8603-37bd8be2b0ae
# ╟─34aaa44a-15af-47d7-af12-5ede12f75255
# ╟─113840b3-dce7-4f0c-bd93-0e1bb76c6a4e
# ╟─10adcba6-ce5d-4a5f-8740-789ccb9c6825
# ╟─f6141719-2ff5-45af-8878-7fc681da5679
# ╟─87151526-4525-49e2-908b-780caa813642
# ╟─651633a7-6172-4bee-8e3c-d53533c9735f
# ╟─29b21dfe-8d2c-4572-b0ee-2836a74b80ea
# ╟─56a95af0-87ae-4b64-b6ce-45f1af84b0f7
# ╟─3a746365-165a-43d9-aaa0-4111778172b5
# ╟─02b1e948-48f0-4451-917c-24fa62e8d465
# ╟─ffab514f-bb9a-472d-8bed-f68259865b9e
# ╟─4a6ec997-fc98-4ed3-84b8-7b88bd040cc8
# ╟─3ffba375-a00b-4084-89d3-a5a2275631ba
# ╟─2a582157-83a3-437c-b7cd-b3996c3515cb
# ╟─632a4db4-971d-4292-a361-c99669533c77
# ╟─b8fd0e80-dd7c-4e52-9773-cd8cd51d1038
# ╟─6a499da5-1896-4856-9c31-e073db0a7398
# ╟─0a1d0171-4ca8-4c35-aab7-2af9085d13fe
# ╟─59945ce5-44df-4af1-a9b1-1ed9b9516d38
# ╟─0b5ab124-7232-4ce0-bac0-a9511f102548
# ╟─f6eb557a-b2ed-498d-b854-bbdc784449de
# ╟─6c973af4-aeab-4bc3-8936-a239d01d00f6
# ╟─d9f8445a-f9b1-49b9-8342-3cca5735f152
# ╟─7295b3e8-e1d9-4c5e-879e-413198fb34d8
# ╟─ec8c9a5a-99ee-4b15-8a66-a23e114d08de
# ╟─b37881f4-acdb-494a-aec1-db484c1ff9b4
# ╟─7b28028b-fc27-4297-abed-44d3afea9fdb
# ╟─9bfa2c3a-aa51-4eec-af3b-b4960e86db13
# ╟─2a5d6ba9-1f78-4385-89ea-34ac810dda09
# ╟─cda7e0c8-58d1-4cb6-8114-d4c22994b62d
# ╟─3f6e13eb-7edd-4f56-a254-358a85fb4cd0
# ╟─6daec259-426d-480d-87c8-496440c32187
# ╟─540d482a-582c-46aa-9319-ac29adcfbd5a
# ╟─cb14e0fb-c864-4a91-bf9d-d97b08177837
# ╟─a6d6da76-c2b2-43f5-a62d-89714db7af70
# ╟─7799abbe-5d61-49e1-8e06-8f56565f87de
# ╟─a025ef1a-e96c-492f-9eb1-4347bce83606
# ╟─48dee03b-9ceb-48fb-a002-28410f3fe1df
# ╟─8e76d171-a6c9-4009-b1b3-de13ea84c758
# ╟─bf2da2d2-d11e-4c6a-9a2b-60e6ca41a437
# ╟─37a3f827-e270-4776-8edc-179130a59cd4
# ╟─610f8dfc-a888-44a8-b292-a250dcedaa7b
# ╟─106abb2b-8b79-4d18-a436-df411160179c
# ╟─32c59be3-efc8-4854-a2c5-aa928accfc9f
# ╟─d1f7b362-0eba-4c4d-9c89-7546f6c9a1f6
# ╟─b0211ec8-ce1c-45aa-b346-e0d0df0b471a
# ╟─8936338a-5ab6-4186-bc9a-0fe91eb384bc
# ╟─aee948c5-28e2-4cb4-b228-bfe3541fbd8d
# ╟─4b3ceee6-0753-427d-a859-6b2570dc24fb
# ╟─2ae650a3-0a84-4a83-8614-eb8f4b72d654
# ╟─bc5a41e4-96ad-4c06-8afb-0e4d4f61f1ec
# ╟─04642132-7128-4c8c-ae4b-11fb4a9419ff
# ╟─c9a2efdf-8aa8-4270-b707-97f8c4e9cca4
# ╟─62edb8f3-8e2c-43d7-9b66-a45f526bf201
# ╟─30b4e409-5ea9-4109-bd74-bc9221fc721a
# ╟─5be338dd-afc9-4e15-b308-53d424450217
# ╟─f3866823-b2fa-4575-8304-23f4c4e16f96
# ╟─12a38212-70db-4155-960d-e989b43e750d
# ╟─7087e085-978a-487e-ba1a-4fc354fd1e5d
# ╟─8bdbf6f0-0deb-4bf3-ab1a-c6edf1af9f44
# ╟─261b196f-fe8f-4e7a-a37d-443ddd11346a
# ╟─6b1c9a67-4365-450b-af34-bd4df85b82b8
# ╟─9c67d367-657e-4d0f-9745-11550f75af01
# ╟─89159376-a626-4a09-960f-8054f36201ee
# ╟─82be8d9d-9275-4ace-ba90-79c848b452ba
# ╟─ee2027ca-fba7-4b11-a596-babba7387ebb
# ╟─b47aa87d-58d7-4000-a53e-105cfcbf6229
# ╟─ec0e0778-b2bf-41bd-8342-40c24c948d9d
# ╟─2d06d398-a61a-4d92-80db-f2e3c857aec8
# ╟─fbbad3be-c5b2-4719-aa1f-2dbe9182f0af
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002