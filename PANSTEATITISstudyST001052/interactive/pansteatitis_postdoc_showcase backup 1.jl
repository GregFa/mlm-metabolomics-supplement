### A Pluto.jl notebook ###
# v0.19.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 148df6fb-feac-4fe2-9ccc-18cb460796bf
begin
	#############
	# Libraries #
	#############
	using CSV, DataFrames, DataFramesMeta, Missings, CategoricalArrays
	using StatsBase, Statistics, MultivariateStats, MatrixLM
	using Random, Distributions, GLM, StatsModels
	using LinearAlgebra, PrettyTables, Latexify
	using FreqTables, Plots, StatsPlots, PlutoUI, Images, FileIO
	using Plots.PlotMeasures, ColorSchemes, RecipesBase
	# using PlotlyBase
	using MetabolomicsWorkbenchAPI
	


	######################
	# External functions #
	######################
	include(joinpath(@__DIR__, "..","src","wrangling_utils.jl" ));
	include(joinpath(@__DIR__, "..","src","utils.jl" ));
	# include(joinpath(@__DIR__, "..","src","demog.jl" ));
	include(joinpath(@__DIR__, "..","src","mLinearModel.jl" ));
	include(joinpath(@__DIR__,"..", "src","myPlots.jl" ));

	myfont = "Helvetica"
	
	# function local_ingredients(path::String)
	# 	# this is from the Julia source code (evalfile in base/loading.jl)
	# 	# but with the modification that it returns the module instead of the last object
	# 	name = Symbol(basename(path))
	# 	m = Module(name)
	# 	Core.eval(m,
	#         Expr(:toplevel,
	#              :(eval(x) = $(Expr(:core, :eval))($name, x)),
	#              :(include(x) = $(Expr(:top, :include))($name, x)),
	#              :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
	#              :(include($path))))
	# 	m
	# end
	# mymodule = local_ingredients(joinpath(@__DIR__, "..","src","PANSTEATITISstudyST001052.jl" ));
	df_study_info = fetch_study_info("ST001052");
end;

# ╔═╡ d9a9dd61-4cab-4ee0-a769-7dfc45c281b5
PlutoUI.TableOfContents()

# ╔═╡ 34d05aeb-63fc-46c5-aefe-b9f6622e6c9b
md"""
# PANSTEATITIS Analysis Using Matrix Linear Models  
---
**Gregory Farage, Chenhao Zhao, Saunak Sen**    

	gfarage@uthsc.edu / sen@uthsc.edu
    Division of Biostatistics
    Department of Preventive Medicine
    University of Tennessee Health Science Center
    Memphis, TN
	
	chenhao.zhao.gr@dartmouth.edu
	Graduate Program in Quantitative Biomedical Sciences
	Geisel School of Medicine
	Dartmouth College
	Hanover, NH
    
"""

# ╔═╡ a59b3bc2-edaa-11ec-3b2d-fd5bdd0ee1f2
md"""
## Background
$(df_study_info.STUDY_SUMMARY[1])
(Koemel, 2018; Koemel *et al.*, 2019)
## Project Summary

This project seeks to investigate the association between metabolite characteristics (e.g., pathways) and patient characteristics such as sex or age by using Matrix Linear Models.

## Methods
 $(load("../images/matrixlinearmodel.png"))    
     
Matrix linear models provide a framework for studying associations in high-throughput data using bilinear models.

$$Y= X B Z^\prime + E,$$

where
- ``Y`` is the outcome data matrix (metabolites levels)
- ``X`` is a design matrix constructed from individual covariates (phenotype status, sex, age, BMI...)
- ``Z`` is a design matrix constructed from outcome covariates (information about the pathways)
- ``E`` is a random error assumed to have a mean zero, independent across individuals, but possibly correlated across outcomes

$$V(vec(E)) = \Sigma \otimes I.$$

"""

# ╔═╡ 58cdac80-d63b-4c3e-9193-d507a0da9be6
md"""
## Data Analysis
"""

# ╔═╡ 92b67799-b650-4f1f-b6b0-70f5a95dd8e1
md"""
>This data is available at the NIH Common Fund's National Metabolomics Data Repository (NMDR) website, the Metabolomics Workbench, [https://www.metabolomicsworkbench.org](https://www.metabolomicsworkbench.org), where it has been assigned Project ID PR000705. The data can be accessed directly via it's Project DOI: [10.21228/M8JH5X](http://dx.doi.org/10.21228/M8JH5X) This work is supported by NIH grant, U2C- DK119886.
"""

# ╔═╡ de5b8917-915c-4c17-8d2c-d55f06822a3f
begin
###############
# Individuals #
###############

fileIndividuals = joinpath(@__DIR__,"..","data","processed",
							"ST001052_ClinicalCovariates.csv");
dfInd = CSV.read(fileIndividuals, DataFrame);
# dfInd = select(dfInd, Not([:Fibrosis,:Ballooning]))
# keep complete cases
dfInd = dfInd[findall(completecases(dfInd)), :];

# names of the categorical variables
vIndivCatNames = ["Group", "Gender"]

#########################
# Metabolites responses #
#########################

# Get metabolite file
fileMeta = joinpath(@__DIR__,"..","data","processed","nl2_Meta.csv");
dfMet = CSV.read(fileMeta, DataFrame);
	

###########################
# Metabolites annotations #
###########################	

# Get reference metabolite file
fileRef = joinpath(@__DIR__,"..","data","processed","refMeta.csv");
dfRef = CSV.read(fileRef, DataFrame);
	
# Sort according to metabolite_ID 
sort!(dfRef, [:MetaboliteID]);
	
# Keep only references according to metabolites in the response dataframe
# Get metabolite ID from the outcome dataframe
dfMetID = DataFrame(MetaboliteID = names(dfMet)[2:end]);
# Use the join function to filter the reference dataframe
dfRef = rightjoin(dfRef,dfMetID, on = :MetaboliteID);

# Group 
dfRef.super_class = String.(replace(dfRef.super_class, missing=>"NA"))
dfRef.sub_class = String.(replace(dfRef.sub_class, missing=>"NA"))
dfRef.Class = String.(replace(dfRef.Class, missing=>"NA"))
dfRef.main_class = String.(replace(dfRef.main_class, missing=>"NA"))	
dfRef.sub_class = replace.(dfRef.sub_class, "O-PC"=>"PC",	"O-PS"=>"PS")
	
	
###########################
# Triglycerides responses #
###########################
	
# select only triglyceride metabolites in dfRef
dfRefTG = filter(row -> row.Class == "TG",dfRef)
	
# sort dfRefTG according to MetaboliteID 
sort!(dfRefTG, [:MetaboliteID]);
	
# select only triglycerides metabolites in dfMet 
dfMetTG = copy(dfMet[:, vcat([:SampleID], Symbol.(dfRefTG.MetaboliteID))])

funStandardize!(dfMetTG, tocenter = true)
	
# get the outcome matrix 
mYTG = Matrix(dfMetTG[:,2:end]);

# get the number of columns in the outcome matrix
mTG = size(mYTG, 2);


# No need here to standardize since it is already done in preprocessing
funStandardize!(dfMet, tocenter = true)

# Order metabolite_ID columns
dfMet = dfMet[!, vcat([:SampleID], Symbol.(sort(names(dfMet)[2:end])))]

# get the outcome matrix 
mY = Matrix(dfMet[:,2:end]);

# get number of columns in the outcome matrix
m = size(mY, 2);
	

end;

# ╔═╡ 9da0827e-aec0-4981-9139-aee445104a04
md"""
### Individual Characteristics
"""

# ╔═╡ ae4fd58c-cd3d-4b7d-8c29-dd84e2a103c9
md"""
The clinical dataset contains *n*= $(size(dfInd, 1)) participants. The following table contains  the clinical data dictionary:
"""

# ╔═╡ 427acceb-45ce-4807-a516-bf4858c5f71e
begin
	fileClinicalDict = joinpath(@__DIR__,"..","data","processed", "ClinicalDataDictionary.csv");
	dfClinicalDict = CSV.read(fileClinicalDict, DataFrame);
	latexify(dfClinicalDict; env=:mdtable, latex=false)
end

# ╔═╡ 4097b5c7-4c10-4fce-b169-ccf0e3e5a7dd
md"""
\* *Notes:*
*Vet score, PCV color, and histological score determine if a tilapia is healthy or pansteatitis-affected.*     
"""

# ╔═╡ 5fd0e864-748e-44eb-99e3-61914860261c
md"""
The following table presents the demographics for the tilapias:
"""

# ╔═╡ 96741af6-faa9-40ef-85e2-8647199ed6ab
begin 
	fileDemog = joinpath(@__DIR__,"..","data","processed",
							"demog.csv");
	dfDemographic = CSV.read(fileDemog, DataFrame);
	latexify(dfDemographic; env=:mdtable, latex=false)
end

# ╔═╡ 1f367dfd-8642-4c2c-806b-42388725f0cf
md"""
\* *Notes:*
*continuous variables show mean(standard deviation); categorical variables show total.*     
"""

# ╔═╡ f679bf30-78a6-48c0-a17d-391a0e185289
md"""
### Metabolites Characteristics
"""

# ╔═╡ 3e5e2657-253c-4f96-a5f3-3aea07485478
md"""
#### Preprocessing
We preprocessed the metabolomic datasets through the following steps: (1) imputation was not needed, (2) we performed a normalization based on the probabilistic quotient normalization (Dieterle *et al.*, 2006) method, (3) we applied a log2-transformation to make data more symmetric and homoscedastic.
"""

# ╔═╡ 6902142b-a049-4724-aa1b-ee5b167c0cd6
md"""
#### Dataset Overview 
The metabolomic dataset contains *m*=$(size(dfMet, 2)-1) metabolites. According to their attributes, each metabolite may belong to one of the $(length(unique(dfRef.super_class))-1) identified classes or to the untargeted class, named "NA". It is also possible to sub-classified them based on  $(length(unique(dfRef.sub_class))-1) sub-classes and one untargeted sub-class, tagged as "NA". The following table presents the $(length(unique(dfRef.main_class))) "super classes" and their ID tags: 
"""

# ╔═╡ 859546bf-7508-45ab-b10e-412314b6d753
begin 
	# freq table
	vFreq = freqtable(dfRef.super_class)
	dfFreq = DataFrame(Class = names(vFreq)[1], Count = vFreq)
	# join dataframes
	dfFreq =leftjoin(rename(sort(unique(dfRef[:,[:super_class]])),
					 [:super_class].=> [:Class]), dfFreq,
					  on = :Class)
	# display table
	latexify(dfFreq; env=:mdtable, latex=false)
end

# ╔═╡ 521b36b1-965a-483e-b67c-b78af8ea129b
md"""
The following table presents the $(length(unique(dfRef.sub_class))) sub-classes and their ID tags: 
"""

# ╔═╡ 5e0c72da-44db-4f92-9002-03682ac3843c
let 
	# freq table
	vFreq = freqtable(dfRef.sub_class)
	dfFreq = DataFrame(SubClass = names(vFreq)[1], Count = vFreq)
	# join dataframes
	dfFreq =leftjoin(rename(unique(dfRef[:,[:sub_class]]),
					  		[:sub_class].=> [:SubClass]),
					dfFreq, on = :SubClass)

	
	fileSubClassDict = joinpath(@__DIR__,"..","data","processed", "SubClassDictionary.csv");
	dfSubClassDict = CSV.read(fileSubClassDict, DataFrame);
	dfFreq = leftjoin(rename(dfSubClassDict, Dict(:Name => "Sub Class")),
		 dfFreq,
		 on = :SubClass)
	
	dfFreq = sort(dfFreq, [:SubClass])
	
	dfFreq = rename(dfFreq, Dict(:SubClass => "ID"))
	latexify(dfFreq; env=:mdtable, latex=false)
end

# ╔═╡ 6952ec52-982d-4a37-ae1a-e7f5f1972fda
md"""
## Modeling Decision

$$Y= X B Z^\prime + E,$$
"""

# ╔═╡ 1f9d797d-5832-4e28-bfe5-056cf3e70a33
md"""
### X matrix
"""

# ╔═╡ 85236909-e107-4aa7-8b19-57892f4610d0
@bind xCovariates MultiCheckBox(vcat(["Intercept"], names(dfInd)[2:end]),
						default = ["Intercept", "Age", "Weight", "Length", "Histological_Score"])

# ╔═╡ 65e58ee6-752d-459c-adc3-81c83bc64d4b
begin 
	vPredictorNames = copy(xCovariates)
	

	if (@isdefined hasInteractions) && (hasInteractions == "Sex Interactions" )
		idxNotSex = findall(xCovariates.!= "Sex" .&& xCovariates.!= "Intercept")
		vPredictorNames[idxNotSex] .= "Sex*".*xCovariates[idxNotSex]
		vPredictorNames = vPredictorNames[Not(findall(vPredictorNames.=="Sex"))]
	else
		vPredictorNames = copy(xCovariates)
		
	end

	if ["Intercept"] ⊆ xCovariates
		vPredictorNames[findall(vPredictorNames .=="Intercept")] .= "1"
	end
	
	frml = join(vPredictorNames, " + ")
	
md"""
Replace `+` by `*` for interaction:

$(@bind tf_frml TextField(default=frml))

"""	

end


# ╔═╡ e57fbd30-8849-4cd5-849f-7b31e4805df3
@bind bttnModel Button("Confirm")

# ╔═╡ 4c37aaae-b669-42e4-8364-c87fec0593eb
frml_c = Ref("");

# ╔═╡ 3b1fb44f-cce0-4e78-9a5a-fd4d811a70e1
begin
bttnModel	
frml_c[] = @eval $(Meta.parse("tf_frml"))
	# tf_frml 
md"""
The @formula design is: $(frml_c[])    
"""		
end

# ╔═╡ 27cee0f5-b86f-404a-9dd8-61bdf72bc43f
begin 
bttnModel
contrasts = Dict(
	:Status => EffectsCoding(base=sort(unique(dfInd.Status))[2]), 
	:Sex => EffectsCoding(base=sort(unique(dfInd.Sex))[1])
)

if frml_c[] != ""
	formulaX = eval(Meta.parse(string("@formula(0 ~ ", frml_c[], ").rhs")))
	if occursin("+", frml_c[]) || occursin("*", frml_c[])
		vCovarNames = collect(string.(formulaX))
	else
		vCovarNames = [string(formulaX)]	
	end
	
	if ["1"] ⊆ vCovarNames
		vCovarNames[findall(vCovarNames .== "1")] .= "Intercept"
	end
	
	idx2change = findall(occursin.("&",vCovarNames))
	vCovarNames[idx2change]  .= replace.(vCovarNames[idx2change], " & " => "Ξ")
	
	mX = modelmatrix(formulaX, dfInd, hints = contrasts)

	
	vFrmlNames = Vector{String}()
	
 	if frml_c[] == "1"
		vFrmlNames = ["(Intercept)"]
	else
		sch = schema(formulaX, dfInd, contrasts)
		vFrmlNames = apply_schema(formulaX, sch) |> coefnames
	end
	function fix_covar_name(s::String)
		s = replace(s, "("=>"", ")"=>"", ": "=>"_")
		s = replace(s, " & " => "Ξ")
	end
	
	vPseudoFrmlNames = fix_covar_name.(vFrmlNames)
	show(vFrmlNames)
end
end

# ╔═╡ 438a108b-1676-46ea-9396-aa71d0320190
begin	
if (@isdefined mX)

	md"""
	Show X matrix $(@bind radio_X CheckBox())
	"""		
end	
end

# ╔═╡ fa290623-12ad-4ba5-a749-ac6531da7c16
begin 
if (@isdefined mX)
	if radio_X
		mX
	else
		md"The size of Pansteatitis X is $(size(mX, 1)) x $(size(mX, 2))."
	end
end
end

# ╔═╡ 6660d119-3a7d-4247-83e2-3585ebb56b0a
md"""
## Metabolites analysis
"""

# ╔═╡ cfcdb796-9088-4746-9a9a-97baa3255ddf
begin
	#######################
	# Plotting attributes #
	#######################
	# myfont = "Helvetica"
	mytitlefontsize = 12 
end;

# ╔═╡ ce038faa-ab1f-4686-b14d-2c7f8ba0a257
md"""
### Association at the metabolite level per super class
"""

# ╔═╡ 38a801e5-b55a-43e6-a3d5-07874d461324
begin
if (@isdefined mX)
	###################
	# Identity Matrix #
	###################
	ZI = Matrix{Float64}(I(m))
	
	CoefZI, CIZI, TstatZI, varZI = getCoefs(mY, mX, ZI);

	dfTstatsZI = DataFrame(hcat(permutedims(TstatZI),names(dfMet)[2:end]),
								 vcat(vPseudoFrmlNames, ["MetaboliteID"]));
	dfTstatsZI = leftjoin(dfTstatsZI, 
			dfRef[:,[:MetaboliteID, :super_class , :sub_class]], on = :MetaboliteID);

	# need to convert for visualization
	for i in Symbol.(vPseudoFrmlNames)
		# dfTstatsZI[:, i] = float.(vec(dfTstatsZI[:, i]))
		dfTstatsZI[!,i] = convert.(Float64, dfTstatsZI[!,i])
	end		

	md"""
	Show Z matrix $(@bind radio_ZI CheckBox())
	"""
	
end
end

# ╔═╡ 2e2f10c8-e70d-43cd-bf5c-1c9dfc0755c1
begin 
if (@isdefined mX)
	if radio_ZI
		ZI
	else
		md"Z matrix is an identity matrix."
	end
end
end

# ╔═╡ 4be72887-8c6d-4812-80f8-6c75e55d73ee
if @isdefined vPseudoFrmlNames
	@bind xCovarFig Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 87f8fc10-c6f0-4487-9394-d22e3b480258
begin
if @isdefined dfTstatsZI
	nameCovarFig = fix_covar_name(xCovarFig)
	gdf = groupby(dfTstatsZI, :super_class);
	dfMeanTst = combine(gdf, Symbol(nameCovarFig) => mean => Symbol(nameCovarFig)) 
	sort!(dfMeanTst, :super_class);
	
	pscatter_sup = eval(Meta.parse("@df dfTstatsZI dotplot(string.(:super_class), :$(nameCovarFig), label = \"Metbolite\", legend = false)"))
	eval(Meta.parse("@df dfMeanTst scatter!(string.(:super_class), :$(nameCovarFig), label = \"Average\", legend = :outerright, xrotation = 20, color = :orange)"))
	hline!([0], color= :red, label = "", bottom_margin = 10mm,
		foreground_color_legend = nothing, # remove box of legend
		xlabel = "Super class", ylabel = string("T-statistics ", nameCovarFig), 
		title = "T-statistics per class",
		titlefontsize = mytitlefontsize,
	)

	savefig(string(
		"../images/pansteatitis_mlmCI_scatter_",
		replace(xCovarFig, " "=> "_", ":"=> "_"),
		"_",
		"super_class",
		".svg")
	)
	
	plot(pscatter_sup)
end
end

# ╔═╡ f5517a21-4885-447c-800f-a065e387e935
md"""
### Association at the super class level
"""

# ╔═╡ f86f116c-fc10-4bb3-80d3-b79d836f1ff6
begin 
if (@isdefined mX)	
	###############
	# Super Class #
	###############
	levelsSup = sort(unique(dfRef.super_class)); 
	# Generate Z matrix
	mZsup = modelmatrix(@formula(y ~ 0 + super_class).rhs, 
						dfRef, 
						hints = Dict(:super_class => StatsModels.FullDummyCoding()));

	# true indicates correct design matrix
	# levelsSup[(mapslices(x ->findall(x .== 1) , mZsup, dims = [2]))[:]] == dfRef.super_class
	CoefZsp, CIZsp, TstatZsp, varZsp = getCoefs(mY, mX, mZsup);	
	
	md"""
	Show Z matrix $(@bind radio_Zsp CheckBox())
	"""		
end	
end

# ╔═╡ 72e71883-fe61-4ce4-9810-bb5db864508d
begin 
if (@isdefined mX)
	if radio_Zsp
		DataFrame(hcat(Int.(mZsup[:,1:end]), dfRef.super_class[:]), vcat(levelsSup, ["Class"]))
	else
		md"Z matrix design is based on super classes."
	end
end
end

# ╔═╡ cfb97bdd-82ad-425e-b85f-2aaa2517d037
if @isdefined vFrmlNames
	@bind xCovarFig_sp Select(vFrmlNames[1:end], default = "Histological_Score")
end

# ╔═╡ 7071ec6c-1196-47ac-906e-f8c9364bbc44
begin
	if @isdefined TstatZsp

		idxCovarsup = findall(vPseudoFrmlNames[1:end] .== fix_covar_name(xCovarFig_sp))
		idx2rmvsup = findall(levelsSup .== "NA")
		# idxCovar = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFig))
		namesZsup = string.(permutedims(levelsSup[1:end]))
		
		psp = confidenceplot(
			vec(permutedims(CoefZsp[idxCovarsup, Not(idx2rmvsup)])), 
			levelsSup[Not(idx2rmvsup)],
			vec(permutedims(CIZsp[idxCovarsup, Not(idx2rmvsup)])),
			fontfamily = "Helvetica",
			xlabel = xCovarFig*" Effect Size", legend = false,
			titlefontsize = mytitlefontsize,
			  )
		
		savefig(string(
			"../images/pansteatitis_mlmCI_",
			replace(xCovarFig_sp, " "=> "_", ":"=> "_"),
			"_",
			"super_class",
			".svg")
		)
		plot(psp)

		
	end
end

# ╔═╡ 582bfd4b-b92f-4cd7-b1f9-96a6bf8dc92e
begin
if (@isdefined TstatZsp)
	# catTstatZsp = zeros(size(TstatZsp,1), size(TstatZsp,2));
	# catTstatZsp[findall(abs.(TstatZsp) .<2)].= 0;
	# catTstatZsp[findall((TstatZsp .>=2) .&& (TstatZsp .<3))].= 1;
	# catTstatZsp[findall((TstatZsp .<=-2) .&& (TstatZsp .>-3))].= -1;
	# catTstatZsp[findall(TstatZsp .>=3)].= 2;
	# catTstatZsp[findall(TstatZsp .<=-3)].= -2;
	# catTstatZsp
	
	# myscheme = [RGB{Float64}(0.792, 0, 0.125),
 #            RGB{Float64}(0.957, 0.647, 0.51),
 #            RGB{Float64}(0.5, 0.5, 0.5),    
 #            RGB{Float64}(0.557, 0.773, 0.871),
 #            RGB{Float64}(0.02, 0.443, 0.69)];
	# labels
	namesTstsX = vFrmlNames
	namesXtbl = reshape(namesTstsX[2:end], 1, length(namesTstsX)-1)
	# clrbarticks = ["T-stats ≤ -3", "-3 < T-stats ≤ -2", "-2 < T-stats < 2", "2 ≤ T-stats < 3", "3 ≤ T-stats"];
	# n_cat = length(clrbarticks)
	# yt = range(0,1,n_cat+1)[1:n_cat] .+ 0.5/n_cat
	# myfont = "Computer Modern" 

	# l = @layout [ a{0.50w} [b{0.1h}; c{0.2w}] ];

	# mycolors =ColorScheme(myscheme)
	# colors = cgrad(mycolors, 5, categorical = true);

	# # PLOT MAIN HEATMAP
	# p1 = mlmheatmap(permutedims(catTstatZsp[2:end,:]), permutedims(TstatZsp[2:end,:]), 
	#     xticks = (collect(1:size(mZsup)[1]), namesXtbl), xrotation = 450,
	#     yticks = (collect(1:size(mZsup)[2]), levelsSup), tickfontfamily = myfont ,
	#     color = colors, cbar = false,
	#     annotationargs = (10, "Arial", :lightgrey), 
	#     linecolor = :white , linewidth = 2);

	# # PLOT GAP 
	# p2 = plot([NaN], lims=(0,1), framestyle=:none, legend=false);

	# # ANNOTATE COLOR BAR TITLE
	# annotate!(0.5, -3.02, text("T-statistics", 10, myfont))
	
	# # PLOT HEATMAP COLOR BAR
	# xx = range(0,1,100)
	# zz = zero(xx)' .+ xx
	# p3 = Plots.heatmap(xx, xx, zz, ticks=false, ratio=3, legend=false, 
	#             fc=colors, lims=(0,1), framestyle=:box, right_margin=5mm);
	
	# # ANNOTATE COLOR BAR AXIS
	# [annotate!(2.15, yi, text(ti, 7, myfont)) for (yi,ti) in zip(yt,clrbarticks)]
	
	# # PLOT MLMHEATMAP
	# plot(p1, p2, p3, layout=l, margins=2mm, size= (700, 550))
end
end;

# ╔═╡ 52509df3-e5ec-405f-be6b-1b87fe068cee
md"""
### Association at the metabolite level per sub-class
"""

# ╔═╡ c1766507-c1e7-4259-a60e-c206a941d359
begin 
if (@isdefined mX)
	#############
	# Sub Class #
	#############
	levelsSub = sort(unique(dfRef.sub_class));

	# Generate Z matrix
	mZsub = modelmatrix(@formula(y ~ 0 + sub_class).rhs, 
						dfRef, 
						hints = Dict(:sub_class => StatsModels.FullDummyCoding()));	
	# true indicates correct design matrix
	# levelsSub[(mapslices(x ->findall(x .== 1) , mZsub, dims = [2]))[:]] == dfRef.sub_class
	CoefZsb, CIZsb, TstatZsb, varZsb = getCoefs(mY, mX, mZsub);

	##################################################
	# Adjust idx by omitting groups with less than 5 #
	##################################################

	# create copd dataframe with the unique SubClassID
	dfPan = select(dfRef, :sub_class);
	# select subpathway with more or equal 5 counts
	dfPan2 = filter(row -> row.nrow > 4,  combine(groupby(dfPan[:, :], [:sub_class]), nrow))

	# select indices for sub_class
	idxpan = findall(x -> [x] ⊆ dfPan2.sub_class, levelsSub)
	
	md"""
	Show Z matrix $(@bind radio_Zsb CheckBox())
	"""		
end
end

# ╔═╡ e63ba87c-3792-4904-b410-bdce82b8a312
begin 
if (@isdefined mX)
	if radio_Zsb
		DataFrame(hcat(Int.(mZsub[:,1:end]), dfRef.super_class[:]), vcat(levelsSub, ["Class"]))
	else
		md"Z matrix design is based on super classes."
	end
end
end

# ╔═╡ 2d648181-e5ab-49f6-b1cd-4db29bc41b54
if @isdefined vPseudoFrmlNames
	@bind xCovarFigsub Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ d161fce0-14dd-44db-ba83-a6f58f4d9bdf
# begin
# 	if @isdefined TstatZsb
# 		# namesX = permutedims(vCovarNames[(vCovarNames[1] == "Intercept" ? 2 : 1):end])
# 		namesZsub = string.(permutedims(levelsSub[1:end]))
# 		plot(transpose(TstatZsb[idxCovar,1:end]), # [:, 1:2]
# 		    label= namesX, xlabel= "Sub class", ylabel = "T-statistics",
# 		    legend =:outerright, lw = 2, marker = :circle, grid = false,
# 		    xticks = (collect(1:size(mZsub)[2]), namesZsub),xrotation = 45,
# 			size = (800,550),
# 			# size = (650,550),
			
# 		    foreground_color_legend = nothing, # remove box of legend   
# 		    bottom_margin = 10mm, format = :svg,
# 		    title = string("T-statistics of Coefficients Estimates") )
# 		hline!([2], color= :red, label = "")
# 		hline!([-2], color= :red, label = "")
# 	end
# end

# ╔═╡ 325c11a3-9f25-4470-bdbb-a5dd837fb94e
begin
if @isdefined dfTstatsZI
	nameCovarFigsub = fix_covar_name(xCovarFigsub)
	gdf_sub = groupby(dfTstatsZI, :sub_class);
	dfMeanTst_sub = combine(gdf_sub, Symbol(nameCovarFigsub) => mean => Symbol(nameCovarFigsub)) 
	sort!(dfMeanTst_sub, :sub_class);
	
	pscatter_sub = eval(Meta.parse("@df dfTstatsZI dotplot(string.(:sub_class), :$(nameCovarFigsub), label = \"Metbolite\", legend = false)"))
	eval(Meta.parse("@df dfMeanTst_sub scatter!(string.(:sub_class), :$(nameCovarFigsub), label = \"Average\", legend = :outerright, xrotation = 20, color = :orange)"))
	hline!([0], color= :red, label = "", bottom_margin = 10mm,
		fontfamily = "Helvetica",
		foreground_color_legend = nothing, # remove box of legend
		xlabel = "Super class", ylabel = string("T-statistics ", xCovarFigsub), 
		title = "T-statistics per class",
		titlefontsize = mytitlefontsize,
	)

	savefig(string(
		"../images/pansteatitis_mlmCI_scatter_",
		replace(xCovarFigsub, " "=> "_", ":"=> "_"),
		"_",
		"sub_class",
		".svg")
	)
	plot(pscatter_sub)
end
end

# ╔═╡ 202b9efe-a41a-4ade-929f-c719a4a17abd
md"""
### Association at the sub class level
"""

# ╔═╡ f6af34b4-6818-4150-8f9f-48330f2fc4af
if @isdefined vPseudoFrmlNames
	@bind xCovarFig_sb Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 42892fea-5bb0-4525-aa8b-acbde875083d
begin
	if @isdefined TstatZsb
		
		idxCovarsub = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFig_sb))
		
		psb = confidenceplot(vec(permutedims(CoefZsb[idxCovarsub, idxpan])), 
		    levelsSub[idxpan],
		    vec(permutedims(CIZsb[idxCovarsub,idxpan])),
		    xlabel = xCovarFig_sb*" Effect Size", legend = false,
		    fontfamily = myfont,
			titlefontsize = mytitlefontsize,
			   )

		savefig(string(
			"../images/pansteatitis_mlmCI_",
			replace(xCovarFig_sb, " "=> "_", ":"=> "_"),
			"_",
			"sub_class",
			".svg")
		)
		plot(psb)
	end
end

# ╔═╡ 40b3e02a-6866-4bdd-8270-1a60f2c818bb
# if @isdefined(TstatZsb)
# 	mlmheatmap(permutedims(TstatZsb[2:end, :]), permutedims(TstatZsb[2:end,:]), 
# 		xticks = (collect(1:size(mZsub)[1]), namesXtbl), xrotation = 25,
# 		yticks = (collect(1:size(mZsub)[2]), levelsSub), tickfontfamily = myfont ,
# 		color = cgrad(:bluesreds,[0.1, 0.3, 0.7, 0.9], alpha = 0.8), cbar = true,
# 		clims = (-3, 3), colorbar_title = "T-statistics",
# 		annotationargs = (10, "Arial", :lightgrey), 
# 		linecolor = :white , linewidth = 2, margins=8mm)
# end

# ╔═╡ e9d1dc37-ffdb-4971-9864-11016fd2ba03
# begin
# if (@isdefined TstatZsb)
# 	catTstatZsb = zeros(size(TstatZsb,1), size(TstatZsb,2));
# 	catTstatZsb[findall(abs.(TstatZsb) .<2)].= 0;
# 	catTstatZsb[findall((TstatZsb .>=2) .&& (TstatZsb .<3))].= 1;
# 	catTstatZsb[findall((TstatZsb .<=-2) .&& (TstatZsb .>-3))].= -1;
# 	catTstatZsb[findall(TstatZsb .>=3)].= 2;
# 	catTstatZsb[findall(TstatZsb .<=-3)].= -2;
# 	catTstatZsb
	
# # 	myscheme = [RGB{Float64}(0.792, 0, 0.125),
# #             RGB{Float64}(0.957, 0.647, 0.51),
# #             RGB{Float64}(0.5, 0.5, 0.5),    
# #             RGB{Float64}(0.557, 0.773, 0.871),
# #             RGB{Float64}(0.02, 0.443, 0.69)];
# 	# labels
# 	namesTstsXsb = vFrmlNames
# 	namesXtblsb = reshape(namesTstsXsb[2:end], 1, length(namesTstsXsb)-1)
# # 	clrbarticks = ["T-stats ≤ -3", "-3 < T-stats ≤ -2", "-2 < T-stats < 2", "2 ≤ T-stats < 3", "3 ≤ T-stats"];
# # 	n_cat = length(clrbarticks)
# # 	yt = range(0,1,n_cat+1)[1:n_cat] .+ 0.5/n_cat
# # 	myfont = "Computer Modern" 

# # 	l = @layout [ a{0.50w} [b{0.1h}; c{0.2w}] ];

# # 	mycolors =ColorScheme(myscheme)
# # 	colors = cgrad(mycolors, 5, categorical = true);

# 	# PLOT MAIN HEATMAP
# 	p1sb = mlmheatmap(permutedims(catTstatZsb[2:end,:]), permutedims(TstatZsb[2:end,:]), 
# 	    xticks = (collect(1:size(mZsub)[1]), namesXtblsb), xrotation = 450,
# 	    yticks = (collect(1:size(mZsub)[2]), levelsSub), tickfontfamily = myfont ,
# 	    color = colors, cbar = false,
# 	    annotationargs = (10, "Arial", :lightgrey), 
# 	    linecolor = :white , linewidth = 2);

# 	# PLOT GAP 
# # 	p2 = plot([NaN], lims=(0,1), framestyle=:none, legend=false);

# 	# ANNOTATE COLOR BAR TITLE
# 	# annotate!(0.5, -3.02, text("T-statistics", 10, myfont))
	
# 	# PLOT HEATMAP COLOR BAR
# # 	xx = range(0,1,100)
# # 	zz = zero(xx)' .+ xx
# # 	p3 = Plots.heatmap(xx, xx, zz, ticks=false, ratio=3, legend=false, 
# # 	            fc=colors, lims=(0,1), framestyle=:box, right_margin=24mm);
	
# 	# ANNOTATE COLOR BAR AXIS
# 	# [annotate!(2.15, yi, text(ti, 7, myfont)) for (yi,ti) in zip(yt,clrbarticks)]
	
# 	# PLOT MLMHEATMAP
# 	plot(p1sb, p2, p3, layout=l, margins=2mm)
# end
# end

# ╔═╡ 220d5244-78fe-4629-a8e3-e2388cc275c6
md"""
## Triglycerides Analysis
"""

# ╔═╡ 1dd31d34-eb68-4950-91bf-8913613b1fbe
Plots.histogram2d(dfRefTG.Total_DB, dfRefTG.Total_C,
			c= :matter, aspectratio = 0.35, xlim = [0,18],
			nbins =18,
			colorbar_title = "Number of Triglycerides",
			xlabel= "Total Double Bonds", ylabel =  "Total Carbons",
			title = string("Number of ", "Triglycerides", " by the number of carbon \n atoms and the number of unsaturated bonds \n"),
			fontfamily = myfont,
			titlefontsize = mytitlefontsize,
			size = (500, 500),
)

# ╔═╡ 860aed82-4296-4ead-81f7-6f0bf1832b35
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTG Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 11163cf5-1681-44c5-a3a1-ae415e9ce494
begin
if (@isdefined mX)
	###################
	# Identity Matrix #
	###################
	ZItg = Matrix{Float64}(I(mTG))
	
	CoefZItg, CIZItg, TstatZItg, varZItg = getCoefs(mYTG, mX, ZItg);

	dfTstatsZItg = DataFrame(hcat(permutedims(TstatZItg),names(dfMetTG)[2:end]),
								 vcat(vPseudoFrmlNames, ["MetaboliteID"]));
	dfTstatsZItg = leftjoin(dfTstatsZItg, 
			dfRefTG[:,[:MetaboliteID, :Total_C , :Total_DB]], on = :MetaboliteID);

	# need to convert for visualization
	for i in Symbol.(vPseudoFrmlNames)
		# dfTstatsZI[:, i] = float.(vec(dfTstatsZI[:, i]))
		dfTstatsZItg[!,i] = convert.(Float64, dfTstatsZItg[!,i])
	end				
	
	# #############################################################
	# # Carbon chain adjusted by Unstauration level - Categorical #
	# #############################################################
	# # dfRefTG₂ = copy(dfRefTG)
	# # levelsC_DB = vcat(["Intercept", "CC_≤_50", "50_<_CC_≤_55"], #"CC_>_55"],
	# # 					levelsDB[1:end-1]);
	# dfRefTG3 = copy(dfRefTG₂)
	# dfRefTG3.Total_DB[findall(dfRefTG₂.Total_DB.<=2)] .=1;
	
	# dfRefTG3.Total_DB[findall(dfRefTG3.Total_DB.>2 .&& dfRefTG3.Total_DB.<=5)] .=2;
	# dfRefTG3.Total_DB[findall(dfRefTG3.Total_DB.>5)] .=3;

	# levelsC_DBcat = ["Intercept", "50_<_CC_≤_55","CC_>_55", "2_<_DB_≤_5","DB_>_5"];

	# # Generate Z matrix
	# mZcdb_cat = modelmatrix(@formula(y ~ 1 + Total_DB*Total_C).rhs, 
	# 					dfRefTG3, 
	# 					hints = Dict(:Total_C => StatsModels.DummyCoding(base =1),
	# 								 :Total_DB => StatsModels.DummyCoding(base = 1)));
	
	# CoefZcdb_cat, CIZcdb_cat, TstatZcdb_cat, varZcdb_cat = getCoefs(mYTG, mX, mZcdb_cat);

	
end
end;

# ╔═╡ c090c950-08cf-4b3f-a06d-e96f51b1b52c
md"""
### Association at the triglycerides level
"""

# ╔═╡ 4a8da8e8-deb8-46b8-8436-47cf555e6be3
md"""
### Association at the unstaturation level
"""

# ╔═╡ d917dd88-b30c-4f51-8b43-2980d3657c43
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGdb Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 0a562001-d0ee-415d-b8d7-3bb9dbd8c39e
begin
if (@isdefined xCovarFigTG)	
	a = 0.25
	namesXtg = xCovarFigTG
	idxCovarTG = findall(vPseudoFrmlNames.==fix_covar_name(namesXtg))
	zlim2 = maximum(abs.(TstatZItg[idxCovarTG,:]))
	ptg_scatter = Plots.scatter(
		dfRefTG.Total_DB .+ rand(Uniform(-a,a),length(dfRefTG.Total_DB)),
		dfRefTG.Total_C,
		zcolor = permutedims(TstatZItg[idxCovarTG,:]), colorbar = true, m = (:bluesreds, 0.8), 
		colorbar_title = string("\nT-statistics of ",namesXtg),
		xticks=collect(0:1:length(unique(dfRefTG.Total_DB))-1), clims = (-zlim2, zlim2), 
		xlabel= "Total Double Bonds", ylabel =  "Total Carbons",
		legend = false, grid = false, tickfontsize=8, 
		fontfamily = "Helvetica",
		right_margin = 10mm,  format = :svg)
		# title = string("Total Carbons according to Total Double Bonds \n with colorbar based on T-statistics of\n ", namesXtg) )
	# savefig(string("../../images/scatterTstats_",namesX[idxRow],".svg"))

	
	savefig(string(
		"../images/pansteatitis_mlmScatter_TG_",
		replace(xCovarFigTGdb, " "=> "_", ":"=> "_"),
		"_",
		"db",
		".svg")
	)
	plot(ptg_scatter)
end
end

# ╔═╡ 4b18f3ec-e53d-4732-899e-d54cd0b00d5c
begin
if (@isdefined TstatZItg)		
	# a = 0.25
	# namesXtg = xCovarFigTG
	# idxCovarTG = findall(vPseudoFrmlNames.==fix_covar_name(namesXtg))
	# zlim2 = maximum(abs.(TstatZItg[idxCovarTG,:]))
	p_db = Plots.scatter(dfRefTG.Total_DB .+ rand(Uniform(-a,a),length(dfRefTG.Total_DB)),
		permutedims(TstatZItg[idxCovarTG,:]),
		# zcolor = permutedims(TstatZItg[idxCovarTG,:]), colorbar = true, m = (:bluesreds, 0.8), 
		colorbar_title = string("\nT-statistics of ",namesXtg),
		xticks=collect(0:1:length(unique(dfRefTG.Total_DB))-1), clims = (-zlim2, zlim2), 
		xlabel= "Total Double Bonds", ylabel =  "T-statistics",
		legend = false, grid = false, tickfontsize=8,
		fontfamily = "Helvetica",
		right_margin = 10mm,  format = :svg);

	p_c = Plots.scatter(dfRefTG.Total_C .+ rand(Uniform(-a,a),length(dfRefTG.Total_C)),
		permutedims(TstatZItg[idxCovarTG,:]),
		# zcolor = permutedims(TstatZItg[idxCovarTG,:]), colorbar = true, m = (:bluesreds, 0.8), 
		colorbar_title = string("\nT-statistics of ",namesXtg),
		xticks=minimum(dfRefTG.Total_C):1:maximum(dfRefTG.Total_C),
		clims = (-zlim2, zlim2), 
		xlabel= "Total Length Carbon", ylabel =  "T-statistics",
		legend = false, grid = false, tickfontsize=8, 
		fontfamily = "Helvetica",
		right_margin = 10mm,  format = :svg);
	
	l = @layout [ a{0.50h}; b{0.5h} ];
	plot(p_db, p_c, layout=l, margins=2mm, size= (700, 550))
	
end
end

# ╔═╡ a59a2f68-6352-4559-b19e-f865fab8203f
begin
if (@isdefined mX)	
	################################
	# Reduced Double-bond TG Class #
	################################
	dfRefTG₂ = copy(dfRefTG)
	# max_db = 12
	# dfRefTG₂.Total_DB[findall(dfRefTG₂.Total_DB.>=max_db)] .=max_db;
	# levelsDB = string.(sort(unique(dfRefTG₂.Total_DB)));

	vbrks_db = [0,3,6,9,12]#collect(0:12);
	lvls_db = vcat(
		string.(vbrks_db[1:end-1]).*" ≤ Double Bonds < ".*string.(vbrks_db[2:end]),
		"Double Bonds > ".*string.(vbrks_db[end]));
	vcatdb = cut(
				dfRefTG.Total_DB, vbrks_db; 
				labels = lvls_db, 
				extend = true)

	dfRefTG₂.Total_DB_cat = vcatdb
  

	# Generate Z matrix
	mZdb_cat = modelmatrix(@formula(y ~ 0 + Total_DB_cat).rhs, 
						dfRefTG₂, 
						hints = Dict(:Total_DB_cat => StatsModels.FullDummyCoding()));
	# true indicates correct design matrix
	# levelsDB[(mapslices(x ->findall(x .== 1) , mZdb, dims = [2]))[:]] == string.(dfRefTG₂.Total_DB)

	
	
	# levelsDB = "Double Bond = ".*levelsDB;
	# levelsDB[end] = "Double Bond ≥ "*string(max_db);

	levelsDB = lvls_db;
	CoefZdb, CIZdb, TstatZdb, varZdb = getCoefs(mYTG, mX, mZdb_cat);
end
end;


# ╔═╡ db4b76c4-a11c-4233-85e5-c57fd77dbf46
begin
	if @isdefined TstatZdb
		
		idxCovarTGdb = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFigTGdb))
		namesZdb = string.(permutedims(levelsDB[1:end]))
		ptg_db = confidenceplot(
				vec(permutedims(CoefZdb[idxCovarTGdb,:])), 
				vec(namesZdb),
				vec(permutedims(CIZdb[idxCovarTGdb,:])),
				fontfamily = myfont,
				xlabel = xCovarFigTGdb*" Effect Size", legend = false,
		)

		savefig(string(
			"../images/pansteatitis_mlmCI_TG_",
			replace(xCovarFigTGdb, " "=> "_", ":"=> "_"),
			"_",
			"db",
			".svg")
		)
		plot(ptg_db)
	end
end

# ╔═╡ 4907447c-063b-4dd7-b209-0f44cfe032ae
Plots.bar(
	freqtable(string.(vcatdb)), 
	xticks = (collect(1:5),namesZdb), 
	rotation = 45,
	legend = false,
	bottom_margin = 10mm
)

# ╔═╡ 361a8430-411e-43d5-8195-7b5e3a8ae954
md"""
### Association at the carbon chain level
"""

# ╔═╡ a3d9ab2b-4b75-43a7-868e-8cacd268da15
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGc Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ a6c5c97e-c3e3-422c-8719-1003ac153837
begin
if (@isdefined mX)	
	#########################
	# Carbon chain TG Class #
	#########################
	# # dfRefTG₃ = copy(dfRefTG)
	# dfRefTG₂.Total_C = copy(dfRefTG.Total_C)
	# dfRefTG₂.Total_C[findall(dfRefTG₂.Total_C.<50)] .= 1;
	# dfRefTG₂.Total_C[findall(dfRefTG₂.Total_C.>=50 .&& dfRefTG₂.Total_C.<55)] .= 2;
	# dfRefTG₂.Total_C[findall(dfRefTG₂.Total_C.>=55 .&& dfRefTG₂.Total_C.<60)] .= 3;
	# dfRefTG₂.Total_C[findall(dfRefTG₂.Total_C.>=60 .&& dfRefTG₂.Total_C.<65)] .= 4;
	# dfRefTG₂.Total_C[findall(dfRefTG₂.Total_C.>=65)] .= 5;

	# # # levelsC = ["CC_≤_50", "50_<_CC_≤_55", "55_<_CC_≤_60", "60_<_CC_≤_65","CC_>_65"];
	# levelsC = ["Total Carbon ≤ 50", "50 < Total Carbon ≤ 55","55 < Total Carbon ≤ 60","60 < Total Carbon ≤ 65","Total Carbon > 65"];

	vbrks_c = [0,50,55,60,65]#collect(0:12);
	lvls_c = vcat(
		string.(vbrks_c[1:end-1]).*" ≤ Total Carbon < ".*string.(vbrks_c[2:end]),
		"Total Carbon > ".*string.(vbrks_c[end]));
	vcatc = cut(
				dfRefTG.Total_C, vbrks_c; 
				labels = lvls_c, 
				extend = true)

	dfRefTG₂.Total_C_cat = vcatc
	levelsC = lvls_c
	# Generate Z matrix
	mZc_cat = modelmatrix(@formula(y ~ 0 + Total_C_cat).rhs, 
						dfRefTG₂, 
						hints = Dict(
								:Total_C_cat => StatsModels.FullDummyCoding()
						)
		);
	# mZc2 = modelmatrix(@formula(y ~ 0 + Total_C).rhs, 
	# 				dfRefTG₂, 
	# 				hints = Dict(
	# 						:Total_C => StatsModels.FullDummyCoding()
	# 				)
	# );

	CoefZc, CIZc, TstatZc, varZc = getCoefs(mYTG, mX, mZc_cat);
end
end;

# ╔═╡ ab027000-fc16-4e4e-85b0-c928f6d32772
begin
	if @isdefined TstatZc
		
		idxCovarTGc = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFigTGc))
		# namesZc = string.(permutedims(levelsC[1:end]))
		ptg_c = confidenceplot(
			vec(permutedims(CoefZc[idxCovarTGc,:])), 
			levelsC,
			vec(permutedims(CIZc[idxCovarTGc,:])),
			xlabel = xCovarFigTGc*" Effect Size", legend = false,
			fontfamily = myfont,
		)

		savefig(string(
			"../images/pansteatitis_mlmCI_TG_",
			replace(xCovarFigTGc, " "=> "_", ":"=> "_"),
			"_",
			"c",
			".svg")
		)
		plot(ptg_c)
	end
end

# ╔═╡ 0d12f436-29a0-447a-b4b7-27211976211a
Plots.bar(
	freqtable(string.(vcatc)), 
	xticks = (collect(1:5),lvls_c), 
	rotation = 45,
	legend = false,
	bottom_margin = 10mm
)

# ╔═╡ 814866a7-bc8a-406c-ae3c-80a0ade42d12
md"""
### Association at the carbon and unsaturation level (categorical)
"""

# ╔═╡ fdeece47-c239-42d5-b06e-7e289edaef72
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGcdbcat Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 0dbce4ff-028a-4734-a490-aeef7ac949c6
begin
if (@isdefined mX)	
	#########################
	# Carbon chain TG Class #
	#########################
	dfRefTG₃ = copy(dfRefTG)
	dfRefTG₃.Total_C[findall(dfRefTG₃.Total_C.<=50)] .= 1;
	dfRefTG₃.Total_C[findall(dfRefTG₃.Total_C.>50 .&& dfRefTG₃.Total_C.<=55)] .= 2;
	dfRefTG₃.Total_C[findall(dfRefTG₃.Total_C.>55 .&& dfRefTG₃.Total_C.<=60)] .= 3;
	dfRefTG₃.Total_C[findall(dfRefTG₃.Total_C.>60 .&& dfRefTG₃.Total_C.<=65)] .= 4;
	dfRefTG₃.Total_C[findall(dfRefTG₃.Total_C.>65)] .= 5;

	dfRefTG₃.Total_DB[findall(dfRefTG₃.Total_DB.<=3)] .= 1;
	dfRefTG₃.Total_DB[findall(dfRefTG₃.Total_DB.>3 .&& dfRefTG₃.Total_DB.<=6)] .= 2;
	dfRefTG₃.Total_DB[findall(dfRefTG₃.Total_DB.>6 .&& dfRefTG₃.Total_DB.<=11)] .= 3;
	# dfRefTG₃.Total_DB[findall(dfRefTG₃.Total_DB.>60 .&& dfRefTG₃.Total_C.<=65)] .= 4;
	dfRefTG₃.Total_DB[findall(dfRefTG₃.Total_DB.>11)] .= 4;


	levelsCdbcat = ["Intercept", "50 < Total Carbon ≤ 55","55 < Total Carbon ≤ 60","60 < Total Carbon ≤ 65", "Total Carbon > 65",
		"3 < Double Bond ≤ 6", "6 < Double Bond ≤ 11", "Double Bond > 11"];

	dfRefTG₃.Total_DB = dfRefTG₃.Total_DB;#.*1.0;
	# dfRefTG₃.Total_C = string.(dfRefTG₃.Total_C)
	# Generate Z matrix


	




	
	mZcdbcat = modelmatrix(@formula(y ~ 1 + Total_C +Total_DB).rhs, 
						dfRefTG₃, 
						hints = Dict(
								:Total_C => StatsModels.DummyCoding(base = 1),
								:Total_DB => StatsModels.DummyCoding(base = 1)
						)
		);


	mZcdb_cat = modelmatrix(@formula(y ~ 1 + Total_C_cat + Total_DB_cat).rhs, 
						dfRefTG₂, 
						hints = Dict(
								:Total_C_cat => StatsModels.DummyCoding(base = levels(dfRefTG₂.Total_C_cat)[1]),
							    :Total_DB_cat => StatsModels.DummyCoding(base = 	levels(dfRefTG₂.Total_DB_cat)[1])
						)
		);

	levelsCdb_cat = vcat(["Intercept"], 
						 levels(dfRefTG₂.Total_C_cat)[2:end],
						 levels(dfRefTG₂.Total_DB_cat)[2:end])

	

	CoefZcdbcat, CIZcdbcat, TstatZcdbcat, varZcdbcat = getCoefs(mYTG, mX, mZcdb_cat);
end
end;

# ╔═╡ 7ac60758-524d-4ac6-9f13-addfb4d099d7
levelsCdb_cat

# ╔═╡ c0bb0798-d2b3-482e-9710-4551eb86b0e4
begin
	if @isdefined TstatZc
		
		idxCovarTGcdbcat = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFigTGcdbcat))
		# namesZc = string.(permutedims(levelsC[1:end]))


		# Join results
	    namesZtcdb = vcat(levelsC[2:end], levelsDB[2:end]) 
	    CoefZtcdb  = hcat(CoefZc[:, 2:end], CoefZdb[:, 2:end])
	    CIZtcdb    = hcat(CIZc[:, 2:end] , CIZdb[:, 2:end])
	    TstatZtcdb = hcat(TstatZc[:, 2:end] , TstatZdb[:, 2:end])
	    varZtcdb   = hcat(varZc[:, 2:end] , varZdb[:, 2:end])
	
		p_ci_tcdb_unadjusted = confidenceplot(
					vec(CoefZtcdb[idxCovarTGcdbcat, :]),	
					namesZtcdb,
					vec(CIZtcdb[idxCovarTGcdbcat, :]),
					xlabel = xCovarFigTGcdbcat*" Effect Size", legend = false,
					fontfamily = myfont,
					title = string("Unadjusted") , 
					titlefontsize = mytitlefontsize,
					yaxis = true, 
					left_margin = 10mm,
					# size=(800,700),
					)

		
		ptg_cdbcat_adjusted = confidenceplot(
				vec(permutedims(CoefZcdbcat[idxCovarTGcdbcat, 2:end])), 
				levelsCdb_cat[2:end],
				vec(permutedims(CIZcdbcat[idxCovarTGcdbcat, 2:end])),
				xlabel = xCovarFigTGcdbcat*" Effect Size", 
				title = string("Adjusted") , 
				legend = false,
				fontfamily = myfont,
				yaxis = false, 
	            left_margin = -20mm,
		)

		p_ua = plot(p_ci_tcdb_unadjusted, ptg_cdbcat_adjusted, size = (750, 500)) 

		
		# savefig(string(
		# 	"../images/pansteatitis_mlmCI_TG_",
		# 	replace(xCovarFigTGcdbcat, " "=> "_", ":"=> "_"),
		# 	"_",
		# 	"cdbcat",
		# 	".svg")
		# )
		# plot(p_ua)
	end
end

# ╔═╡ b288574e-ac2f-4570-b880-d4341a9022e3
confidenceplot(
				vec(permutedims(CoefZcdbcat[idxCovarTGcdbcat, 2:end])), 
				levelsCdb_cat[2:end],
				vec(permutedims(CIZcdbcat[idxCovarTGcdbcat, 2:end])),
				xlabel = xCovarFigTGcdbcat*" Effect Size", 
				title = string("Adjusted") , 
				legend = false,
				fontfamily = myfont,
				yaxis = false, 
	            # left_margin = -20mm,
		)

# ╔═╡ eb317298-087f-4e44-975e-d8249485119f
plot(vec(permutedims(CoefZcdbcat[idxCovarTGcdbcat, 2:end])),
yaxis = false)

# ╔═╡ 55c501ac-f208-413a-a772-574d818fa6a1
md"""
### Association at the carbon number and unsaturation level as continuous variables
"""

# ╔═╡ c3a2675b-3276-48f4-8a9a-3851194c1679
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGcdb Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ d9f26d07-639d-4ef6-ad46-fe7e5dc725da
begin
if (@isdefined mX)	
	###############################################
	# Carbon chain adjusted by Unstauration level #
	###############################################
	# Generate Z matrix
	# mZcdb = modelmatrix(@formula(y ~ 1 + Total_DB*Total_C).rhs, 
	# 					dfRefTG3, 
	# 					hints = Dict(:Total_C => StatsModels.DummyCoding(base =1),
	# 								 :Total_DB => StatsModels.DummyCoding(base = 1)));
	dfRefTG4 = select(dfRefTG, [:Total_DB, :Total_C])
	dfRefTG4.Total_C = float.(dfRefTG4.Total_C)./(maximum(dfRefTG4.Total_C))#5 # per six unit Carbon
	dfRefTG4.Total_DB = float.(dfRefTG4.Total_DB)./(maximum(dfRefTG4.Total_DB))

	levelsC_DB = ["Intercept", "Double Bond","Carbon Chain", "DoubleBondΞCarbonChain"];

	# Generate Z matrix
	mZcdb = modelmatrix(@formula(y ~ 1 + Total_DB*Total_C).rhs, 
						dfRefTG4);

	CoefZcdb, CIZcdb, TstatZcdb, varZcdb = getCoefs(mYTG, mX, mZcdb);
end
end;	

# ╔═╡ 350f90cb-f9cf-43b8-a517-b9d5e2cbd269
begin
	if @isdefined TstatZc
		
		idxCovarTGcdb = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFigTGcdb))
		namesZcdb = string.(permutedims(levelsC_DB[2:end]))
		ptg_cdb = confidenceplot(vec(permutedims(CoefZcdb[idxCovarTGcdb,2:end])), 
				   vec(namesZcdb),
				   vec(permutedims(CIZcdb[idxCovarTGcdb,2:end])),
				   xlabel = xCovarFigTGcdb*" Effect Size", legend = false,
				   fontfamily = myfont,
		)
		
		savefig(string(
			"../images/pansteatitis_mlmCI_TG_",
			replace(xCovarFigTGcdb, " "=> "_", ":"=> "_"),
			"_",
			"cdb",
			".svg")
		)
		plot(ptg_cdb)		
	end
end

# ╔═╡ c5173a1d-6084-4408-a372-d14fd6ab8012
md"""
### Association at the carbon number and oxidation
"""

# ╔═╡ 8d66e321-a180-488c-af46-c31c00d416d1
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGoxc Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 3dde17b1-3129-4b9e-b540-0e1f50301e9f
begin
if (@isdefined xCovarFigTGoxc)	
	# a = 0.25
	namesXtgoxc = xCovarFigTGoxc
	idxCovarTGox = findall(vPseudoFrmlNames.==fix_covar_name(namesXtgoxc))
	zlim3 = maximum(abs.(TstatZItg[idxCovarTG,:]))
	Plots.scatter((dfRefTG.Oxidized .*1) .+ rand(Uniform(-a,a),length(dfRefTG.Oxidized)),
		dfRefTG.Total_C,
		zcolor = permutedims(TstatZItg[idxCovarTGox,:]), colorbar = true, m = (:bluesreds, 0.8), 
		colorbar_title = string("\nT-statistics of ",namesXtgoxc),
		xticks=collect(0:1:length(unique(dfRefTG.Oxidized))-1), clims = (-zlim3, zlim3), 
		xlabel= "Oxidation status", ylabel =  "Total Carbons",
		legend = false, grid = false, tickfontsize=8, 
		fontfamily = "Helvetica",
		right_margin = 10mm,  format = :svg)
		# title = string("Total Carbons according to Total Double Bonds \n with colorbar based on T-statistics of\n ", namesXtg) )
	# savefig(string("../../images/scatterTstats_",namesX[idxRow],".svg"))
end
end

# ╔═╡ c77dd4fa-8c3a-43c8-a98e-9a3f3744703f
begin
if (@isdefined mX)	
	###############################################
	# Carbon chain adjusted by Unstauration level #
	###############################################
	# Generate Z matrix
	# mZcdb = modelmatrix(@formula(y ~ 1 + Total_DB*Total_C).rhs, 
	# 					dfRefTG3, 
	# 					hints = Dict(:Total_C => StatsModels.DummyCoding(base =1),
	# 								 :Total_DB => StatsModels.DummyCoding(base = 1)));
	dfRefTGox = select(dfRefTG, [:Oxidized, :Total_C])
	dfRefTGox.Total_C = float.(dfRefTGox.Total_C)./5 # per six unit Carbon
	dfRefTGox.Oxidized = Int.(dfRefTGox.Oxidized)

	# Generate Z matrix
	formulaOxc = @formula(y ~ 1 + Oxidized * Total_C).rhs;
	sch_oxc = schema(formulaOxc, dfRefTGox, 
					 Dict(:Oxidized => StatsModels.DummyCoding(base =0)))
	vFrmlNames_oxc = apply_schema(formulaOxc, sch_oxc) |> coefnames
	
	mZoxc = modelmatrix(formulaOxc, 
						dfRefTGox,
						hints = Dict(:Oxidized => StatsModels.DummyCoding(base =0))
						);
	levelsOx_C = vFrmlNames_oxc;

	CoefZoxc, CIZoxc, TstatZoxc, varZoxc = getCoefs(mYTG, mX, mZoxc);
end
end;	

# ╔═╡ bf7b998d-4a64-4c33-8243-f146f4e798d6
begin
	if @isdefined TstatZoxc
		
		idxCovarTGoxc = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFigTGoxc))
		namesZoxc = string.(permutedims(levelsOx_C[2:end]))
		ptg_oxc = confidenceplot(vec(permutedims(CoefZoxc[idxCovarTGoxc,2:end])), 
				   vec(namesZoxc),
				   vec(permutedims(CIZoxc[idxCovarTGoxc,2:end])),
				   xlabel = xCovarFigTGoxc*" Effect Size", legend = false,
				   fontfamily = myfont,
		)
		
		savefig(string(
			"../images/pansteatitis_mlmCI_TG_",
			replace(xCovarFigTGoxc, " "=> "_", ":"=> "_"),
			"_",
			"oxc",
			".svg")
		)
		plot(ptg_oxc)		
	end
end

# ╔═╡ d2695be2-1604-4fd6-8a8a-63f13061a7f7
md"""
### Association at the carbon number and plasmanyl type
"""

# ╔═╡ 60d5fbd6-f5f1-46cb-8031-7abdfa9cc040
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGplm Select(vFrmlNames, default = "Histological_Score")
end

# ╔═╡ 8af320d1-d465-4cf2-98a3-caa2e48e113f
begin
if (@isdefined xCovarFigTGplm)	
	# a = 0.25
	namesXtgplm = xCovarFigTGoxc
	idxCovarTGplm = findall(vPseudoFrmlNames.==fix_covar_name(namesXtgplm))
	zlim4 = maximum(abs.(TstatZItg[idxCovarTGplm,:]))
	Plots.scatter((dfRefTG.Plasmanyl .*1) .+ rand(Uniform(-a,a),length(dfRefTG.Plasmanyl)),
		dfRefTG.Total_C,
		zcolor = permutedims(TstatZItg[idxCovarTGplm,:]), colorbar = true, m = (:bluesreds, 0.8), 
		colorbar_title = string("\nT-statistics of ",namesXtgplm),
		xticks=collect(0:1:length(unique(dfRefTG.Plasmanyl))-1), clims = (-zlim4, zlim4), 
		xlabel= "Plasmanyl status", ylabel =  "Total Carbons",
		legend = false, grid = false, tickfontsize=8, 
		fontfamily = "Helvetica",
		right_margin = 10mm,  format = :svg)
		# title = string("Total Carbons according to Total Double Bonds \n with colorbar based on T-statistics of\n ", namesXtg) )
	# savefig(string("../../images/scatterTstats_",namesX[idxRow],".svg"))
end
end

# ╔═╡ 74c81e18-925a-44ef-9252-7c7a97503099
begin
if (@isdefined mX)	
	###############################################
	# Carbon chain adjusted by Unstauration level #
	###############################################
	# Generate Z matrix
	# mZcdb = modelmatrix(@formula(y ~ 1 + Total_DB*Total_C).rhs, 
	# 					dfRefTG3, 
	# 					hints = Dict(:Total_C => StatsModels.DummyCoding(base =1),
	# 								 :Total_DB => StatsModels.DummyCoding(base = 1)));
	dfRefTGplm = select(dfRefTG, [:Plasmanyl, :Total_C])
	dfRefTGplm.Total_C = float.(dfRefTGplm.Total_C)./5 # per six unit Carbon
	dfRefTGplm.Plasmanyl = Int.(dfRefTGplm.Plasmanyl)

	# Generate Z matrix
	formulaPlm = @formula(y ~ 1 + Plasmanyl * Total_C).rhs;
	sch_plm = schema(formulaPlm, dfRefTGplm, 
					 Dict(:Plasmanyl => StatsModels.DummyCoding(base =0)))
	vFrmlNames_plm = apply_schema(formulaPlm, sch_plm) |> coefnames
	
	mZplm = modelmatrix(formulaPlm, 
						dfRefTGplm,
						hints = Dict(:Plasmanyl => StatsModels.DummyCoding(base =0))
						);
	levelsPlm_C = vFrmlNames_plm;

	CoefZplm, CIZplm, TstatZplm, varZplm = getCoefs(mYTG, mX, mZplm);
end
end;

# ╔═╡ 0e6d9607-9ec0-45f1-aca7-83d3873d2a3b
begin
	if @isdefined TstatZplm
		
		# idxCovarTGplm = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFigTGplm))
		namesZplm = string.(permutedims(levelsPlm_C[2:end]))
		ptg_plm = confidenceplot(vec(permutedims(CoefZplm[idxCovarTGplm,2:end])), 
				   vec(namesZplm),
				   vec(permutedims(CIZplm[idxCovarTGplm,2:end])),
				   xlabel = xCovarFigTGplm*" Effect Size", legend = false,
				   fontfamily = myfont,
		)
		
		savefig(string(
			"../images/pansteatitis_mlmCI_TG_",
			replace(xCovarFigTGplm, " "=> "_", ":"=> "_"),
			"_",
			"oxc",
			".svg")
		)
		plot(ptg_plm)		
	end
end

# ╔═╡ 0bea76b5-5fc2-4a33-9143-5b10c026f335
md"""
## References

>[Koelmel, J. P. (2018). Lipidomics for wildlife disease etiology and biomarker discovery: a case study of pansteatitis outbreak in South Africa (part-I), NIH Common Fund's National Metabolomics Data Repository (NMDR) website, the Metabolomics Workbench, https://www.metabolomicsworkbench.org, Summary of Study ST001052, doi: 10.21228/M8JH5X](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001052&StudyType=MS&ResultType=1)

>[Koelmel, J. P., Ulmer, C. Z., Fogelson, S., Jones, C. M., Botha, H., Bangma, J. T., Guillette, T. C., Luus-Powell, W. J., Sara, J. R., Smit, W. J., Albert, K., Miller, H. A., Guillette, M. P., Olsen, B. C., Cochran, J. A., Garrett, T. J., Yost, R. A., & Bowden, J. A. (2019). Lipidomics for wildlife disease etiology and biomarker discovery: a case study of pansteatitis outbreak in South Africa. Metabolomics : Official journal of the Metabolomic Society, 15(3), 38. https://doi.org/10.1007/s11306-019-1490-9](https://doi.org/10.1007/s11306-019-1490-9)
	
>[Dieterle F, Ross A, Schlotterbeck G, Senn H. Probabilistic Quotient Normalization as Robust Method to Account for Dilution of Complex Biological Mixtures. Application in 1H NMR Metabonomics. Analytical Chemistry. 2006;78: 4281–4290. doi:10.1021/ac051632c](https://doi.org/10.1021/ac051632c)

>[Hastie, T., Tibshirani, R., Sherlock, G., Eisen, M., Brown, P. and Botstein, D., Imputing Missing Data for Gene Expression Arrays, Stanford University Statistics Department Technical report (1999), http://www-stat.stanford.edu/~hastie/Papers/missing.pdf Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing value estimation methods for DNA microarrays BIOINFORMATICS Vol. 17 no. 6, 2001 Pages 520-525](http://www-stat.stanford.edu/~hastie/Papers/missing.pdf)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataFramesMeta = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
FreqTables = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MatrixLM = "37290134-6146-11e9-0c71-a5c489be1f53"
MetabolomicsWorkbenchAPI = "19b29032-9db8-4baa-af7b-2b362e62b3d7"
Missings = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.4"
CategoricalArrays = "~0.10.6"
ColorSchemes = "~3.19.0"
DataFrames = "~1.3.4"
DataFramesMeta = "~0.11.0"
Distributions = "~0.25.62"
FileIO = "~1.14.0"
FreqTables = "~0.4.5"
GLM = "~1.8.0"
Images = "~0.25.2"
Latexify = "~0.15.15"
MatrixLM = "~0.1.3"
MetabolomicsWorkbenchAPI = "~0.2.0"
Missings = "~1.0.2"
MultivariateStats = "~0.9.1"
Plots = "~1.30.0"
PlutoUI = "~0.7.39"
PrettyTables = "~1.3.1"
RecipesBase = "~1.2.1"
StatsBase = "~0.33.16"
StatsModels = "~0.6.30"
StatsPlots = "~0.14.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "c07f92c229d63458896cd8f7508ab64ccfad2d82"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5f5a975d996026a8dd877c35fe26a7b8179c02ba"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.6"

[[deps.Chain]]
git-tree-sha1 = "339237319ef4712e6e5df7758d0bccddf5c237d9"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.10"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "OrderedCollections", "Reexport"]
git-tree-sha1 = "f1d89a07475dc4b03c08543d1c6b4b2945f33eca"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.11.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "6180800cebb409d7eeef8b2a9a562107b9705be5"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.67"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreqTables]]
deps = ["CategoricalArrays", "Missings", "NamedArrays", "Tables"]
git-tree-sha1 = "488ad2dab30fd2727ee65451f790c81ed454666d"
uuid = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
version = "0.4.5"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "039118892476c2bf045a43b88fcb75ed566000ff"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2d908286d120c584abbe7621756c341707096ba4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.2+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "db5c7e27c0d46fd824d470a3c32a4fc6c935fa96"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b1798a4a6b9aafb530f8f0c4a7b2eb5501e2f2a3"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.16"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "124626988534986113cfd876e3093e4a03890f58"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "Statistics"]
git-tree-sha1 = "0c703732335a75e683aec7fdfc6d5d1ebd7c596f"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.3"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "b563cf9ae75a635592fc73d3eb78b86220e55bd8"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.6"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "03d1301b7ec885b266c0f816f338368c6c0b81bd"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "57af5939800bce15980bddd2426912c4f83012d8"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "f1572de22c866dc92aea032bc89c2b137cbddd6a"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.10.0"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixLM]]
deps = ["DataFrames", "Distributed", "GLM", "LinearAlgebra", "Random", "SharedArrays", "Statistics", "Test"]
git-tree-sha1 = "742d34bc491fa02269dd019a0fed8f7e40e549c0"
uuid = "37290134-6146-11e9-0c71-a5c489be1f53"
version = "0.1.3"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[deps.MetabolomicsWorkbenchAPI]]
deps = ["CSV", "DataFrames", "HTTP", "JSON3"]
git-tree-sha1 = "16b38874ec66128c3de5a9d08c1fa6157fadb390"
uuid = "19b29032-9db8-4baa-af7b-2b362e62b3d7"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e925a64b8585aa9f4e3047b8d2cdc3f0e79fd4e4"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.16"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d0a61518267b44a70427c0b690b5e993a4f5fe01"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.30.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra", "Random"]
git-tree-sha1 = "b327e4db3f2202a4efafe7569fcbe409106a1f75"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.5.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "3177100077c68060d63dd71aec209373c3ec339b"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "85bc4b051546db130aeb1e8a696f1da6d4497200"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.5"

[[deps.StaticArraysCore]]
git-tree-sha1 = "5b413a57dd3cea38497d745ce088ac8592fbb5be"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.1.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "f8ba54b202c77622a713e25e7616d618308b34d3"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.31"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "fcf41697256f2b759de9380a7e8196d6516f0310"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.0"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "4ad90ab2bbfdddcae329cba59dab4a8cdfac3832"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.7"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─d9a9dd61-4cab-4ee0-a769-7dfc45c281b5
# ╟─34d05aeb-63fc-46c5-aefe-b9f6622e6c9b
# ╠═148df6fb-feac-4fe2-9ccc-18cb460796bf
# ╟─a59b3bc2-edaa-11ec-3b2d-fd5bdd0ee1f2
# ╟─58cdac80-d63b-4c3e-9193-d507a0da9be6
# ╟─92b67799-b650-4f1f-b6b0-70f5a95dd8e1
# ╠═de5b8917-915c-4c17-8d2c-d55f06822a3f
# ╟─9da0827e-aec0-4981-9139-aee445104a04
# ╟─ae4fd58c-cd3d-4b7d-8c29-dd84e2a103c9
# ╟─427acceb-45ce-4807-a516-bf4858c5f71e
# ╟─4097b5c7-4c10-4fce-b169-ccf0e3e5a7dd
# ╟─5fd0e864-748e-44eb-99e3-61914860261c
# ╟─96741af6-faa9-40ef-85e2-8647199ed6ab
# ╟─1f367dfd-8642-4c2c-806b-42388725f0cf
# ╟─f679bf30-78a6-48c0-a17d-391a0e185289
# ╟─3e5e2657-253c-4f96-a5f3-3aea07485478
# ╟─6902142b-a049-4724-aa1b-ee5b167c0cd6
# ╟─859546bf-7508-45ab-b10e-412314b6d753
# ╟─521b36b1-965a-483e-b67c-b78af8ea129b
# ╟─5e0c72da-44db-4f92-9002-03682ac3843c
# ╟─6952ec52-982d-4a37-ae1a-e7f5f1972fda
# ╟─1f9d797d-5832-4e28-bfe5-056cf3e70a33
# ╟─85236909-e107-4aa7-8b19-57892f4610d0
# ╟─65e58ee6-752d-459c-adc3-81c83bc64d4b
# ╟─e57fbd30-8849-4cd5-849f-7b31e4805df3
# ╟─4c37aaae-b669-42e4-8364-c87fec0593eb
# ╟─3b1fb44f-cce0-4e78-9a5a-fd4d811a70e1
# ╟─27cee0f5-b86f-404a-9dd8-61bdf72bc43f
# ╟─438a108b-1676-46ea-9396-aa71d0320190
# ╟─fa290623-12ad-4ba5-a749-ac6531da7c16
# ╟─6660d119-3a7d-4247-83e2-3585ebb56b0a
# ╟─cfcdb796-9088-4746-9a9a-97baa3255ddf
# ╟─ce038faa-ab1f-4686-b14d-2c7f8ba0a257
# ╟─38a801e5-b55a-43e6-a3d5-07874d461324
# ╟─2e2f10c8-e70d-43cd-bf5c-1c9dfc0755c1
# ╟─4be72887-8c6d-4812-80f8-6c75e55d73ee
# ╟─87f8fc10-c6f0-4487-9394-d22e3b480258
# ╟─f5517a21-4885-447c-800f-a065e387e935
# ╟─f86f116c-fc10-4bb3-80d3-b79d836f1ff6
# ╟─72e71883-fe61-4ce4-9810-bb5db864508d
# ╟─cfb97bdd-82ad-425e-b85f-2aaa2517d037
# ╟─7071ec6c-1196-47ac-906e-f8c9364bbc44
# ╟─582bfd4b-b92f-4cd7-b1f9-96a6bf8dc92e
# ╟─52509df3-e5ec-405f-be6b-1b87fe068cee
# ╟─c1766507-c1e7-4259-a60e-c206a941d359
# ╟─e63ba87c-3792-4904-b410-bdce82b8a312
# ╟─2d648181-e5ab-49f6-b1cd-4db29bc41b54
# ╟─d161fce0-14dd-44db-ba83-a6f58f4d9bdf
# ╟─325c11a3-9f25-4470-bdbb-a5dd837fb94e
# ╟─202b9efe-a41a-4ade-929f-c719a4a17abd
# ╟─f6af34b4-6818-4150-8f9f-48330f2fc4af
# ╟─42892fea-5bb0-4525-aa8b-acbde875083d
# ╟─40b3e02a-6866-4bdd-8270-1a60f2c818bb
# ╟─e9d1dc37-ffdb-4971-9864-11016fd2ba03
# ╟─220d5244-78fe-4629-a8e3-e2388cc275c6
# ╟─1dd31d34-eb68-4950-91bf-8913613b1fbe
# ╟─860aed82-4296-4ead-81f7-6f0bf1832b35
# ╟─11163cf5-1681-44c5-a3a1-ae415e9ce494
# ╟─c090c950-08cf-4b3f-a06d-e96f51b1b52c
# ╟─0a562001-d0ee-415d-b8d7-3bb9dbd8c39e
# ╟─4b18f3ec-e53d-4732-899e-d54cd0b00d5c
# ╟─4a8da8e8-deb8-46b8-8436-47cf555e6be3
# ╟─d917dd88-b30c-4f51-8b43-2980d3657c43
# ╠═a59a2f68-6352-4559-b19e-f865fab8203f
# ╟─db4b76c4-a11c-4233-85e5-c57fd77dbf46
# ╠═4907447c-063b-4dd7-b209-0f44cfe032ae
# ╟─361a8430-411e-43d5-8195-7b5e3a8ae954
# ╟─a3d9ab2b-4b75-43a7-868e-8cacd268da15
# ╠═a6c5c97e-c3e3-422c-8719-1003ac153837
# ╠═ab027000-fc16-4e4e-85b0-c928f6d32772
# ╠═0d12f436-29a0-447a-b4b7-27211976211a
# ╟─814866a7-bc8a-406c-ae3c-80a0ade42d12
# ╠═fdeece47-c239-42d5-b06e-7e289edaef72
# ╠═0dbce4ff-028a-4734-a490-aeef7ac949c6
# ╠═7ac60758-524d-4ac6-9f13-addfb4d099d7
# ╠═c0bb0798-d2b3-482e-9710-4551eb86b0e4
# ╠═b288574e-ac2f-4570-b880-d4341a9022e3
# ╠═eb317298-087f-4e44-975e-d8249485119f
# ╟─55c501ac-f208-413a-a772-574d818fa6a1
# ╟─c3a2675b-3276-48f4-8a9a-3851194c1679
# ╟─d9f26d07-639d-4ef6-ad46-fe7e5dc725da
# ╟─350f90cb-f9cf-43b8-a517-b9d5e2cbd269
# ╟─c5173a1d-6084-4408-a372-d14fd6ab8012
# ╟─8d66e321-a180-488c-af46-c31c00d416d1
# ╟─3dde17b1-3129-4b9e-b540-0e1f50301e9f
# ╟─c77dd4fa-8c3a-43c8-a98e-9a3f3744703f
# ╟─bf7b998d-4a64-4c33-8243-f146f4e798d6
# ╟─d2695be2-1604-4fd6-8a8a-63f13061a7f7
# ╟─60d5fbd6-f5f1-46cb-8031-7abdfa9cc040
# ╟─8af320d1-d465-4cf2-98a3-caa2e48e113f
# ╟─74c81e18-925a-44ef-9252-7c7a97503099
# ╟─0e6d9607-9ec0-45f1-aca7-83d3873d2a3b
# ╟─0bea76b5-5fc2-4a33-9143-5b10c026f335
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
