### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1fc288d0-e38d-11ed-1239-49fc6de01db3
begin
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 4ffdd121-dceb-4e8a-8f84-f21aad3f84f2
begin
	#############
	# Libraries #
	#############

	
	# using CSV, DataFrames, DataFramesMeta, Missings, CategoricalArrays
	# using StatsBase, Statistics, MultivariateStats, MatrixLM
	# using Random, Distributions, GLM, StatsModels
	# using LinearAlgebra, PrettyTables, Latexify
	# using FreqTables, Plots, StatsPlots, PlutoUI, Images, FileIO
	# using Plots.PlotMeasures, ColorSchemes, RecipesBase
	# # using PlotlyBase
	# using MetabolomicsWorkbenchAPI

	using Plots, StatsPlots, Images
	using Plots.PlotMeasures, ColorSchemes, RecipesBase
	using CSV, DataFrames, DataFramesMeta, Missings, CategoricalArrays
	using FreqTables
	using StatsBase, Statistics, Distributions, StatsModels, LinearAlgebra
	using PrettyTables, Latexify
	using MatrixLM, MetabolomicsWorkbenchAPI


	# ######################
	# # External functions #
	# ######################
	include(joinpath(@__DIR__, "..","src","wrangling_utils.jl" ));
	include(joinpath(@__DIR__, "..","src","utils.jl" ));
	# include(joinpath(@__DIR__, "..","src","demog.jl" ));
	include(joinpath(@__DIR__, "..","src","mLinearModel.jl" ));
	include(joinpath(@__DIR__,"..", "src","recipe_plots.jl" ));

	myfont = "Helvetica"
	
	df_study_info = fetch_study_info("ST001052");
end;

# ╔═╡ dd113c37-a08b-4c31-b5bc-7a38d2efdcbe
md"""
# PANSTEATITIS Analysis Using Matrix Linear Models  
---
**Gregory Farage, Chenhao Zhao, Saunak Sen**    

	gfarage@uthsc.edu / sen@uthsc.edu
    Division of Biostatistics
    Department of Preventive Medicine
    University of Tennessee Health Science Center
    Memphis, TN
	
	czhao20@uthsc.edu
    Division of Biostatistics
    Department of Preventive Medicine
    University of Tennessee Health Science Center
    Memphis, TN
    
"""

# ╔═╡ e8e4eb9d-12b7-486d-9b40-7174aba0b072
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

# ╔═╡ da4d2151-6cae-41fc-8966-de03decaf5a8
md"""
## Data Analysis
"""

# ╔═╡ 7c18f869-f0ce-44c6-99d3-c0ab2b801001
md"""
>This data is available at the NIH Common Fund's National Metabolomics Data Repository (NMDR) website, the Metabolomics Workbench, [https://www.metabolomicsworkbench.org](https://www.metabolomicsworkbench.org), where it has been assigned Project ID PR000705. The data can be accessed directly via it's Project DOI: [10.21228/M8JH5X](http://dx.doi.org/10.21228/M8JH5X) This work is supported by NIH grant, U2C- DK119886.
"""

# ╔═╡ bc027ee4-9912-480a-82ab-7b7f00d29403
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

# ╔═╡ a77a3328-9fb7-4bfb-b576-00fe281b826e
md"""
### Individual Characteristics
"""

# ╔═╡ 7292bdd6-1e43-464e-829a-013809f06f46
md"""
The clinical dataset contains *n*= $(size(dfInd, 1)) participants. The following table contains  the clinical data dictionary:
"""

# ╔═╡ af2399d5-6679-4529-a382-b3b0b9601882
begin
	fileClinicalDict = joinpath(@__DIR__,"..","data","processed", "ClinicalDataDictionary.csv");
	dfClinicalDict = CSV.read(fileClinicalDict, DataFrame);
	latexify(dfClinicalDict; env=:mdtable, latex=false)
end

# ╔═╡ d5276e0c-8b29-4efa-b1c6-d080658a7a3c
md"""
\* *Notes:*
*Vet score, PCV color, and histological score determine if a tilapia is healthy or pansteatitis-affected.*     
"""

# ╔═╡ edc7f20a-5477-410b-b6d2-64795d0e29dd
md"""
The following table presents the demographics for the tilapias:
"""

# ╔═╡ 6d84ebd3-6949-42ff-9b06-f7fd4e7c8f0d
begin 
	fileDemog = joinpath(@__DIR__,"..","data","processed",
							"demog.csv");
	dfDemographic = CSV.read(fileDemog, DataFrame);
	latexify(dfDemographic; env=:mdtable, latex=false)
end

# ╔═╡ 5ee27ca3-a0ee-42a5-a0f2-0eddee47f81a
# begin 
# 	dfg = groupby(dfInd, :Status);
# 	histogram(dfg[1].Age)
# 	# histogram()
# end

# ╔═╡ adb0c520-3229-48c5-ac4d-5fb2e0bf8093
# combine(dfg, :Age => mean, :Age => std)

# ╔═╡ 473b48ae-b5c6-4de4-b776-6b664bf46082
# histogram(dfg[2].Age)

# ╔═╡ ef386a1a-f755-49fa-80bb-27d5c338d2e3
md"""
### Metabolites Characteristics
"""

# ╔═╡ 1cc20658-6725-4a47-a0be-4e42ad2c80b9
md"""
#### Preprocessing
We preprocessed the metabolomic datasets through the following steps: (1) imputation was not needed, (2) we performed a normalization based on the probabilistic quotient normalization (Dieterle *et al.*, 2006) method, (3) we applied a log2-transformation to make data more symmetric and homoscedastic.
"""

# ╔═╡ e41ff6d5-f332-4601-9092-7d182bcd76d2
md"""
#### Dataset Overview 
The metabolomic dataset contains *m*=$(size(dfMet, 2)-1) metabolites. According to their attributes, each metabolite may belong to one of the $(length(unique(dfRef.super_class))-1) identified classes or to the untargeted class, named "NA". It is also possible to sub-classified them based on  $(length(unique(dfRef.sub_class))-1) sub-classes and one untargeted sub-class, tagged as "NA". The following table presents the $(length(unique(dfRef.main_class))) "super classes" and their ID tags: 
"""

# ╔═╡ 3316f2ff-1ca2-43c3-9c59-dc352b75f9a1
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

# ╔═╡ 60e617ba-4ed4-4353-9afe-61fa677319c0
md"""
The following table presents the $(length(unique(dfRef.sub_class))) sub-classes and their ID tags: 
"""

# ╔═╡ fc95c8db-3373-4665-afa5-e79c65c0d738
begin 
	# freq table
	vFreq_sub = freqtable(dfRef.sub_class)
	dfFreq_sub = DataFrame(SubClass = names(vFreq_sub)[1], Count = vFreq_sub)
	# join dataframes
	dfFreq_sub =leftjoin(rename(unique(dfRef[:,[:sub_class]]),
					  		[:sub_class].=> [:SubClass]),
					dfFreq_sub, on = :SubClass)

	
	fileSubClassDict = joinpath(@__DIR__,"..","data","processed", "SubClassDictionary.csv");
	dfSubClassDict = CSV.read(fileSubClassDict, DataFrame);
	dfFreq_sub = leftjoin(rename(dfSubClassDict, Dict(:Name => "Sub Class")),
		 dfFreq_sub,
		 on = :SubClass)
	
	dfFreq_sub = sort(dfFreq_sub, [:SubClass])
	
	dfFreq_sub = rename(dfFreq_sub, Dict(:SubClass => "ID"))
	latexify(dfFreq_sub; env=:mdtable, latex=false)
end

# ╔═╡ 53fc9db0-28d3-4deb-96f0-91a8e1fe6dbc
md"""
## Modeling Decision

$$Y= X B Z^\prime + E,$$
"""

# ╔═╡ 4d665b9d-a085-482f-b60f-670d3cc2d541
md"""
### X matrix
"""

# ╔═╡ 1cf4e218-e1e6-47ad-80a2-94fddedecb10
@bind xCovariates MultiCheckBox(vcat(["Intercept"], names(dfInd)[2:end]),
						default = ["Intercept", "Status", "Sex", "Age", "Weight", "Length"])

# ╔═╡ 7d855e7e-8dff-4de9-b5ee-a7b09d00d978
if ["Histological_Score"] ⊆ xCovariates
	md"Categorize Histological Score $(@bind isHistoScoreCat CheckBox())"
end

# ╔═╡ ff672c36-1722-44d6-9003-0a3a57c05bf6
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


# ╔═╡ b9642a82-4595-4d3b-a240-cc43253fb2a2
@bind bttnModel Button("Confirm")

# ╔═╡ 3f34b8a0-9655-454c-8554-f95821375e90
frml_c = Ref("");

# ╔═╡ 8f39a6ee-fe9b-421e-b4ca-fc88c706f2b9
begin
bttnModel	
frml_c[] = @eval $(Meta.parse("tf_frml"))
	# tf_frml 
md"""
The @formula design is: $(frml_c[])    
"""		
end

# ╔═╡ e62d4343-8261-4c49-94ac-06cb9db67f68
begin 
bttnModel
contrasts = Dict(
	:Status => EffectsCoding(base=sort(unique(dfInd.Status))[2]), 
	:Sex => EffectsCoding(base=sort(unique(dfInd.Sex))[1]),
	# :Histological_Score => StatsModels.FullDummyCoding(),
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

	if ["Histological_Score"] ⊆ vCovarNames
		vCovarNames[findall(vCovarNames .== "1")] .= "Intercept"
	end
	
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

	if (@isdefined isHistoScoreCat)
		if isHistoScoreCat
	
			mHS = modelmatrix(@formula(0 ~ Histological_Score).rhs, 
					dfInd,
					hints = Dict(:Histological_Score => StatsModels.FullDummyCoding()))
			
	
			
			mX = hcat(mX[:,1:end-1], 
				  sum(mHS[:,2:3], dims= 2),
				  sum(mHS[:,4:6], dims= 2))
			vFrmlNames = vcat(vFrmlNames[1:end-1,],
							["Histological_Score: 1_2"],
							["Histological_Score: 3_5"])
		end
	end
	
	vPseudoFrmlNames = fix_covar_name.(vFrmlNames)


	
	show(vFrmlNames)
end
end

# ╔═╡ 59c324c4-8b3d-4d0d-a460-13d1a334c8eb
begin	
if (@isdefined mX)

	md"""
	Show X matrix $(@bind radio_X CheckBox())
	"""		
end	
end

# ╔═╡ ca12f163-518f-4899-8730-0bf5e019f899
begin 
if (@isdefined mX)
	if radio_X
		mX
	else
		md"The size of Pansteatitis X is $(size(mX, 1)) x $(size(mX, 2))."
	end
end
end

# ╔═╡ 372f70bf-8b76-4f8a-af3a-955622fd6424
md"""
## Metabolites analysis
"""

# ╔═╡ 143a5bcc-6f88-42f5-b8f9-c8487a6aa030
begin
	#######################
	# Plotting attributes #
	#######################
	# myfont = "Helvetica"
	mytitlefontsize = 12 
end;

# ╔═╡ a27ebbdd-0a8b-4f11-a0ab-6a41c991ef88
md"""
### Association at the metabolite level per super class
"""

# ╔═╡ df6118c5-c498-413b-a530-9003fa1340fd
begin
if (@isdefined mX)
	###################
	# Identity Matrix #
	###################
	ZI = Matrix{Float64}(I(m))

	if ("(Intercept)" in vFrmlNames)
		CoefZI, CIZI, TstatZI, varZI = getCoefs(
			mY, mX, ZI; 
			hasXIntercept=false, hasZIntercept= false
		);
	else 
		CoefZI, CIZI, TstatZI, varZI = getCoefs(
			mY, mX, ZI; 
			hasXIntercept=true, hasZIntercept= false
		);
	end
	
	# CoefZI, CIZI, TstatZI, varZI = getCoefs(mY, mX, ZI);

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

# ╔═╡ 8da7da20-885d-43c7-a8d9-1b78baedd11d
begin 
if (@isdefined mX)
	if radio_ZI
		ZI
	else
		md"Z matrix is an identity matrix."
	end
end
end

# ╔═╡ 11d017f1-e15f-47df-9e23-d4d9629ebd63
if @isdefined vPseudoFrmlNames
	@bind xCovarFig Select(vFrmlNames, default = "Age")
end

# ╔═╡ 380feaa3-5a52-4803-a561-3f055510ae16
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

# ╔═╡ 5720877a-e485-4a1d-b286-b7572909658c
md"""
### Association at the super class level
"""

# ╔═╡ 52868e72-74b1-4cfc-9ba6-945fe3c6db8f
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

	if ("(Intercept)" in vFrmlNames)
		CoefZsp, CIZsp, TstatZsp, varZsp = getCoefs(
			mY, mX, mZsup; 
			hasXIntercept=false, hasZIntercept= false
		);
	else 
		CoefZsp, CIZsp, TstatZsp, varZsp = getCoefs(
			mY, mX, mZsup; 
			hasXIntercept=true, hasZIntercept= false
		);
	end
	
	# CoefZsp, CIZsp, TstatZsp, varZsp = getCoefs(mY, mX, mZsup);	
	
	md"""
	Show Z matrix $(@bind radio_Zsp CheckBox())
	"""		
end	
end

# ╔═╡ 1a53fd65-8ce5-452f-98a5-129ef8d56329
begin 
if (@isdefined mX)
	if radio_Zsp
		# DataFrame(hcat(Int.(mZsup[:,1:end]), dfRef.super_class[:]), vcat(levelsSup, ["Class"]))
		DataFrame(Int.(mZsup[:,1:end]), levelsSup)
	else
		md"Z matrix design is based on super classes."
	end
end
end

# ╔═╡ 72ab7cdb-f509-48a4-b587-57de058182a8
if @isdefined vFrmlNames
	@bind xCovarFig_sp Select(vFrmlNames[1:end], default = "Age")
end

# ╔═╡ 1bb3bb3a-9ffe-4690-817b-ea83ce9e46b6
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
			titlefontsize = mytitlefontsize,
			xlabel = xCovarFig_sp*" Effect Size", 
			legend = false,
			
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

# ╔═╡ 30f08888-0d6c-463d-afce-d60a7deade83
md"""
### Association at the metabolite level per sub-class
"""

# ╔═╡ 825605ec-7c8e-4452-ba4d-63b7d5204d3a
begin 
if (@isdefined mX)
	#############
	# Sub Class #
	#############
# 	levelsSub = sort(unique(dfRef.sub_class));

# 	# Generate Z matrix
# 	mZsub = modelmatrix(@formula(y ~ 0 + sub_class).rhs, 
# 						dfRef, 
# 						hints = Dict(:sub_class => StatsModels.FullDummyCoding()));	
# 	# true indicates correct design matrix
# 	# levelsSub[(mapslices(x ->findall(x .== 1) , mZsub, dims = [2]))[:]] == dfRef.sub_class
# 	CoefZsb, CIZsb, TstatZsb, varZsb = getCoefs(mY, mX, mZsub);

# 	##################################################
# 	# Adjust idx by omitting groups with less than 5 #
# 	##################################################

# 	# create pansteatitis dataframe with the unique SubClassID
# 	dfPan = select(dfRef, :sub_class);
# 	# select subpathway with more or equal 5 counts
# 	dfPan2 = filter(row -> row.nrow > 4,  combine(groupby(dfPan[:, :], [:sub_class]), nrow))

# 	# select indices for sub_class
# 	idxpan = findall(x -> [x] ⊆ dfPan2.sub_class, levelsSub)
	

	##################################################
	# Adjust idx by omitting groups with less than 5 #
	##################################################

	# create pansteatitis dataframe with the unique SubClassID
	dfPan = select(dfRef, :sub_class)
	# select subpathway with more or equal 5 counts
	dfPan2 = filter(row -> row.nrow > 4,  combine(groupby(dfPan[:, :], [:sub_class]), nrow))

	# select indices for sub_class
	idx_greater_4 = findall(x -> [x] ⊆ dfPan2.sub_class, dfRef.sub_class)
	# sort sub class
	levelsSub = sort(unique(dfRef.sub_class));
	# select indices for sub_class
	idxpan = findall(x -> [x] ⊆ dfPan2.sub_class, levelsSub)

	dfRef_2 = filter(row -> [row.sub_class] ⊆ levelsSub[idxpan], dfRef)

	# Generate Z matrix
	mZsub = modelmatrix(@formula(y ~ 0 + sub_class).rhs, 
						dfRef_2, 
						hints = Dict(:sub_class => StatsModels.FullDummyCoding()));	
	# true indicates correct design matrix
	# levelsSub[(mapslices(x ->findall(x .== 1) , mZsub, dims = [2]))[:]] == dfRef.sub_class

	if ("(Intercept)" in vFrmlNames)
		CoefZsb, CIZsb, TstatZsb, varZsb = getCoefs(
			mY[:, idx_greater_4], mX, mZsub; 
			hasXIntercept=false, hasZIntercept= false
		);
	else 
		CoefZsb, CIZsb, TstatZsb, varZsb = getCoefs(
			mY[:, idx_greater_4], mX, mZsub; 
			hasXIntercept=true, hasZIntercept= false
		);
	end
	
	# CoefZsb, CIZsb, TstatZsb, varZsb = getCoefs(mY[:, idx_greater_4], mX, mZsub);

	md"""
	Show Z matrix $(@bind radio_Zsb CheckBox())
	"""	

end
end

# ╔═╡ 63f2a028-c4a6-4205-bfa5-288a2fcb02c9
if @isdefined vPseudoFrmlNames
	@bind xCovarFigsub Select(vFrmlNames, default = "Age")
end

# ╔═╡ a094cdf1-bd0c-4c31-ac0b-dd453ca8075f
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

# ╔═╡ b9b257e4-61bf-4362-bb7f-39931ac0cc0e
md"""
### Association at the sub class level
"""

# ╔═╡ 86b5ad8d-65de-40f8-aff0-b82c1431712d
if @isdefined vPseudoFrmlNames
	@bind xCovarFig_sb Select(vFrmlNames, default = "Age")
end

# ╔═╡ d0969087-77c8-410d-b2c1-6a4a6a6a316d
begin
	if @isdefined TstatZsb
		
		idxCovarsub = findall(vPseudoFrmlNames .== fix_covar_name(xCovarFig_sb))
		idx_sub = findall((collect(dfFreq_sub.Count) .>=5) .&& 
						  (dfFreq_sub.ID .!= "NA")
		);

		idx_NA_sub = findall(dfFreq_sub.ID .== "NA")
		
		psb = confidenceplot(vec(permutedims(CoefZsb[idxCovarsub, Not(idx_NA_sub)])),#,idxpan])), 
			dfFreq_sub.:("Sub Class")[idx_sub],
		    # levelsSub[idxpan],
		    vec(permutedims(CIZsb[idxCovarsub, Not(idx_NA_sub)])),# idxpan])),
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

# ╔═╡ d0044ffa-40b7-4675-95fc-e280b28a1207
begin 
if (@isdefined mX)
	if radio_Zsb
		# DataFrame(hcat(Int.(mZsub[:,1:end]), dfRef.super_class[:]), vcat(levelsSub, ["Class"]))
		DataFrame(Int.(mZsub[:,1:end]), dfFreq_sub.:("Sub Class")[idx_sub])
	else
		md"Z matrix design is based on sub classes."
	end
end
end

# ╔═╡ 34449f78-6a79-475d-b2cd-ba2beab31ee9
# begin
# 	l2 = @layout [a b]
# 	psp_2 = confidenceplot(
# 		vec(permutedims(CoefZsp[idxCovarsup, Not(idx2rmvsup)])), 
# 		levelsSup[Not(idx2rmvsup)],
# 		vec(permutedims(CIZsp[idxCovarsup, Not(idx2rmvsup)])),
# 		fontfamily = "Helvetica",
# 		titlefontsize = mytitlefontsize,
# 		xlabel = xCovarFig_sp*" Effect Size\n (a)", 
# 		legend = false,
		
# 	)
	
# 	psb_2 = confidenceplot(
# 		vec(permutedims(CoefZsb[idxCovarsub, Not(idx_NA_sub)])),#,idxpan])), 
# 		dfFreq_sub.:("Sub Class")[idx_sub],
# 		# levelsSub[idxpan],
# 		vec(permutedims(CIZsb[idxCovarsub, Not(idx_NA_sub)])),# idxpan])),
# 		xlabel = xCovarFig_sb*" Effect Size\n (b)", legend = false,
# 		fontfamily = myfont,
# 		titlefontsize = mytitlefontsize,
# 		bottommargin = (5,:mm)
# 	)
	
# 	p_test1= plot(psb_2, psp_2, layout = l2, size = (750, 450)) 

# 			savefig(string(
# 			"../images/ptest1",
# 			".svg")
# 		)

# 	plot(p_test1)


# 	p_test2= plot(psp_2, psb_2, layout = l2, size = (750, 440)) 

# 	savefig(string(
# 			"../images/ptest2",
# 			".svg")
# 		)

# 	plot(p_test2)
# end

# ╔═╡ 47bc75f3-a8fd-4878-9dcd-5ce7151c0820
md"""
## Triglycerides Analysis
"""

# ╔═╡ 9020e518-ab75-4827-b7b2-d0f25c11a0dc
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

# ╔═╡ 8903ef6a-eff8-4f9c-8f31-374e1e2533c0
md"""
### Association at the triglycerides level
"""

# ╔═╡ a4363e40-4420-428f-aec8-9808d39f0e08
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTG Select(vFrmlNames, default = "Age")
end

# ╔═╡ bb96c16e-2d89-483e-891d-c636f514e30f
begin
if (@isdefined mX)
	###################
	# Identity Matrix #
	###################
	ZItg = Matrix{Float64}(I(mTG))

	if ("(Intercept)" in vFrmlNames)
		CoefZItg, CIZItg, TstatZItg, varZItg = getCoefs(
			mYTG, mX, ZItg; 
			hasXIntercept=true, hasZIntercept= false
		);
	else 
		CoefZItg, CIZItg, TstatZItg, varZItg = getCoefs(
			mYTG, mX, ZItg; 
			hasXIntercept=false, hasZIntercept= false
		);
	end
		
	# CoefZItg, CIZItg, TstatZItg, varZItg = getCoefs(mYTG, mX, ZItg);

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

# ╔═╡ f578ed91-eec9-4394-8cdc-39f6511a2be9
begin
if (@isdefined xCovarFigTG)	
	a = 0.25
	namesXtg = xCovarFigTG
	idxCovarTG = findall(vPseudoFrmlNames.==fix_covar_name(namesXtg))
	zlim2 = maximum(abs.(TstatZItg[idxCovarTG,:]))
	ptg_scatter = Plots.scatter(
		dfRefTG.Total_DB .+ rand(Uniform(-a,a),length(dfRefTG.Total_DB)),
		dfRefTG.Total_C .+ rand(Uniform(-a,a),length(dfRefTG.Total_C)),
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
		replace(xCovarFigTG, " "=> "_", ":"=> "_"),
		"_",
		"db",
		".svg")
	)
	plot(ptg_scatter)
end
end

# ╔═╡ c228eba0-89d6-40fe-8d5e-94a41627c4c9
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
		left_margin = 10mm,  format = :svg);

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
	plot(p_db, p_c, layout=l, margins=2mm, size= (700, 1200))
	
end
end

# ╔═╡ 50332a44-81c0-4b80-ab42-517b8196294b
md"""
### Association at the unstaturation level
"""

# ╔═╡ 63e8f438-abd0-4ee7-8121-cf557e56dc73
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGdb Select(vFrmlNames, default = "Age")
end

# ╔═╡ a599374a-dfd5-45b4-b7bd-9bb728872b55
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
  
	levelsDB = lvls_db;
	
	# true indicates correct design matrix
	# levelsDB[(mapslices(x ->findall(x .== 1) , mZdb, dims = [2]))[:]] == string.(dfRefTG₂.Total_DB)

	# levelsDB = "Double Bond = ".*levelsDB;
	# levelsDB[end] = "Double Bond ≥ "*string(max_db);
	
	
	# Generate Z matrix
	mZdb_cat = modelmatrix(@formula(y ~ 1 + Total_DB_cat).rhs, 
						dfRefTG₂, 
						hints = Dict(:Total_DB_cat => 
						StatsModels.DummyCoding(;
								base = lvls_db[1],
								levels = lvls_db	
						)));
						# StatsModels.FullDummyCoding()));


	# mZdb_cat = modelmatrix(@formula(y ~ 1 + Total_DB_cat).rhs, 
	# 					dfRefTG₂, 
	# 					hints = Dict(:Total_DB_cat => StatsModels.DummyCoding(base = 			levels(dfRefTG₂.Total_DB_cat)[1])));

	
	if ("(Intercept)" in vFrmlNames)
		CoefZdb, CIZdb, TstatZdb, varZdb = getCoefs(
			mYTG, mX, mZdb_cat; 
			hasXIntercept=true, hasZIntercept= true
		);
	else 
		CoefZdb, CIZdb, TstatZdb, varZdb = getCoefs(
			mYTG, mX, mZdb_cat; 
			hasXIntercept=false, hasZIntercept= false
		);
	end
	
	# CoefZdb, CIZdb, TstatZdb, varZdb = getCoefs(mYTG, mX, mZdb_cat);
end
end;


# ╔═╡ c7fafba4-4c76-4f14-98d6-fd44fe772628
begin 
	# Create a DataFrame from tStats_diff and append column names from df_baseline
	# dfTstatsZI = DataFrame(
	# hcat(permutedims(TstatZI), names(dfY)[1:end]),
	# vcat(["Intercept", "FishOil"], ["lipID"])
	# );
	
	# Join dfTstatsZI with dfRef to add SuperClassID and SubClassID using CHEM_ID1 as the key
	dfTstatsZI_cdb = leftjoin(
	dfTstatsZItg, 
	dfRefTG₂[:, [:MetaboliteID,  :Total_DB_cat]], on = :MetaboliteID
	# dfRefTG[:, [:lipID, :Total_C_cat , :Total_DB_cat]], on = :lipID
	)
	
	# Generate names for covariate figures based on indices
	# nameCovarFig = xCovarFigTGc
	
	# Group data by SuperClassID
	gdf_db = groupby(dfTstatsZI_cdb, :Total_DB_cat);
	
	# Calculate the mean T-statistics for each super class and create a new DataFrame
	dfMeanTst_db = DataFrames.combine(gdf_db, Symbol(xCovarFigTGdb) => mean => Symbol(xCovarFigTGdb)) 
	# Sort the DataFrame by SuperClassID
	sort!(dfMeanTst_db, :Total_DB_cat);
	
	# Create a dot plot of T-statistics by super class
	p_dot_db = eval(Meta.parse("@df dfTstatsZI_cdb dotplot(string.(:Total_DB_cat), :$(xCovarFigTGdb), legend = false, markersize = 4)"))
	# Overlay a scatter plot on the dot plot with mean values
	eval(Meta.parse("@df dfMeanTst_db scatter!(string.(:Total_DB_cat), :$(xCovarFigTGdb), legend = false, color = :orange)"))
	# Add a horizontal line at T=0 for reference
	hline!([0], color= :red, 
	label = "",
	xlabel = "Total DB category", xrotation = 45,
	ylabel = string("T-statistics ", "Treatment"),
	title = "T-statistics per class",
	titlefontsize = mytitlefontsize,
	fontfamily = myfont, grid = false,
	)
	
	# Display the plot
	plot(p_dot_db)
end

# ╔═╡ ce42d8b6-003e-4e5c-83ef-41372a3526b8
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

# ╔═╡ 87f44b92-74a1-4524-b227-a93a3ac895ad
Plots.bar(
	collect(freqtable(string.(vcatdb))),
	xticks = (collect(1:5),namesZdb), 
	rotation = 45,
	legend = false,
	bottom_margin = 10mm
)

# ╔═╡ a46aa97d-679a-4b37-a6c0-a6c5148be92f
md"""
### Association at the carbon chain level
"""

# ╔═╡ 23f880c2-b334-4472-9c93-733ebdc56ecf
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGc Select(vFrmlNames, default = "Age")
end

# ╔═╡ 0f84687b-38b0-4302-99e6-43e3aa2c6443
begin
if (@isdefined mX)	
	#########################
	# Carbon chain TG Class #
	#########################

	vbrks_c = [0,50,55,60,65]
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
	mZc_cat = modelmatrix(@formula(y ~ 1 + Total_C_cat).rhs, 
						dfRefTG₂, 
						hints = Dict(
							:Total_C_cat => StatsModels.DummyCoding(;
								base = lvls_c[1],
								levels = lvls_c
							)));	
							# :Total_C_cat => StatsModels.FullDummyCoding()));

	if ("(Intercept)" in vFrmlNames)
		CoefZc, CIZc, TstatZc, varZc = getCoefs(
			mYTG, mX, mZc_cat; 
			hasXIntercept=true, hasZIntercept= true
		);
	else 
		CoefZc, CIZc, TstatZc, varZc = getCoefs(
			mYTG, mX, mZc_cat; 
			hasXIntercept=false, hasZIntercept= false
		);
	end
	
	# CoefZc, CIZc, TstatZc, varZc = getCoefs(mYTG, mX, mZc_cat);
end
end;

# ╔═╡ dc3046c8-b3e3-454b-8ea2-4b5a1780db6e
begin 
	# Create a DataFrame from tStats_diff and append column names from df_baseline
	# dfTstatsZI = DataFrame(
	# hcat(permutedims(TstatZI), names(dfY)[1:end]),
	# vcat(["Intercept", "FishOil"], ["lipID"])
	# );
	
	# Join dfTstatsZI with dfRef to add SuperClassID and SubClassID using CHEM_ID1 as the key
	dfTstatsZI_c = leftjoin(
	dfTstatsZItg, 
	dfRefTG₂[:, [:MetaboliteID, :Total_C_cat, :Total_DB_cat]], on = :MetaboliteID
	# dfRefTG[:, [:lipID, :Total_C_cat , :Total_DB_cat]], on = :lipID
	)
	
	# Generate names for covariate figures based on indices
	# nameCovarFig = xCovarFigTGc
	
	# Group data by SuperClassID
	gdf_c = groupby(dfTstatsZI_c, :Total_C_cat);
	
	# Calculate the mean T-statistics for each super class and create a new DataFrame
	dfMeanTst_c = DataFrames.combine(gdf_c[1:end], Symbol(xCovarFigTGc) => mean => Symbol(xCovarFigTGc)) 
	# Sort the DataFrame by SuperClassID
	sort!(dfMeanTst_c, :Total_C_cat);
	
	# Create a dot plot of T-statistics by super class
	p_dot = eval(Meta.parse("@df dfTstatsZI_c dotplot(string.(:Total_C_cat), :$(xCovarFigTGc), legend = false, markersize = 4)"))
	# Overlay a scatter plot on the dot plot with mean values
	eval(Meta.parse("@df dfMeanTst_c scatter!(string.(:Total_C_cat), :$(xCovarFigTGc), legend = false, color = :orange)"))
	# Add a horizontal line at T=0 for reference
	hline!([0], color= :red, 
	label = "",
	xlabel = "Total C category", xrotation = 45,
	ylabel = string("T-statistics ", "Treatment"),
	title = "T-statistics per class",
	titlefontsize = mytitlefontsize,
	fontfamily = myfont, grid = false,
	)
	
	# Display the plot
	plot(p_dot)
end

# ╔═╡ 0e510137-2641-43a1-af41-fb1fae61ecdf
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
			".svg"))
		
		plot(ptg_c)
	end
end

# ╔═╡ 07eb16a0-68bf-4cd5-bd8c-2387497e57d6
# Plots.bar(
# 	collect(freqtable(string.(vcatc))), 
# 	xticks = (collect(1:5),lvls_c), 
# 	rotation = 45,
# 	legend = false,
# 	bottom_margin = 10mm
# )

# ╔═╡ f77f6c00-3a8a-45f6-a3d0-801693ea6812
md"""
### Association at the carbon and unsaturation level (categorical)
"""

# ╔═╡ 57ddd596-68b7-43bd-93a8-998d16848f85
if @isdefined vPseudoFrmlNames
	@bind xCovarFigTGcdbcat Select(vFrmlNames, default = "Age")
end

# ╔═╡ e80aee70-62ad-446e-884c-c1785218f9cb
begin
if (@isdefined mX)	
	##########################################
	# Carbon chain and Double Bonds TG Class #
	##########################################

	mZcdb_cat = modelmatrix(@formula(y ~ 1 + Total_C_cat + Total_DB_cat).rhs, 
						dfRefTG₂, 
						hints = Dict(
								:Total_C_cat => StatsModels.DummyCoding(;
									base =	levelsC[1],
									levels = levelsC
								),
							    :Total_DB_cat => StatsModels.DummyCoding(
									base =	levelsDB[1],
									levels = levelsDB
									)));

	levelsCdb_cat = vcat(["Intercept"], 
						 levels(dfRefTG₂.Total_C_cat)[2:end],
						 levels(dfRefTG₂.Total_DB_cat)[2:end])

	
	if ("(Intercept)" in vFrmlNames)
		CoefZcdbcat, CIZcdbcat, TstatZcdbcat, varZcdbcat = getCoefs(
			mYTG, mX, mZcdb_cat; 
			hasXIntercept=true, hasZIntercept= true
		);
	else 
		CoefZcdbcat, CIZcdbcat, TstatZcdbcat, varZcdbcat = getCoefs(
			mYTG, mX, mZcdb_cat; 
			hasXIntercept=false, hasZIntercept= false
		);
	end
	
	# CoefZcdbcat, CIZcdbcat, TstatZcdbcat, varZcdbcat = getCoefs(mYTG, mX, mZcdb_cat);
end
end;

# ╔═╡ 4c237533-1186-4b7f-81a0-8fe3a271c437
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
					xlim = (-0.25,0.25),
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
				xlim = (-0.25,0.25),
				title = string("Adjusted") , 
				legend = false,
				fontfamily = myfont,
				titlefontsize = mytitlefontsize,
				yaxis = false, 
	            left_margin = -20mm,
		)

		p_ua = plot(p_ci_tcdb_unadjusted, ptg_cdbcat_adjusted, size = (750, 440)) 

		
		savefig(string(
			"../images/pansteatitis_mlmCI_TG_",
			replace(xCovarFigTGcdbcat, " "=> "_", ":"=> "_"),
			"_",
			"cdbcat",
			".svg")
		)
		plot(p_ua)
	end
end

# ╔═╡ 7097bd8d-bb77-45fc-a101-c157519abff6
 confidenceplot(
					vec(CoefZtcdb[idxCovarTGcdbcat, :]),	
					namesZtcdb,
					vec(CIZtcdb[idxCovarTGcdbcat, :]),
					xlabel = xCovarFigTGcdbcat*" Effect Size", legend = false,
					# xlim = (-1,1),
					fontfamily = myfont,
					title = string("Unadjusted") , 
					titlefontsize = mytitlefontsize,
					yaxis = true, 
					left_margin = 10mm,
					# size=(800,700),
					)

# ╔═╡ 836efb58-18ff-4a3e-8a55-5b0a4457a13d
confidenceplot(
				vec(permutedims(CoefZcdbcat[idxCovarTGcdbcat, 2:end])), 
				levelsCdb_cat[2:end],
				vec(permutedims(CIZcdbcat[idxCovarTGcdbcat, 2:end])),
				xlabel = xCovarFigTGcdbcat*" Effect Size", 
				# xlim = (-0.25,0.25),
				title = string("Adjusted") , 
				legend = false,
				fontfamily = myfont,
				titlefontsize = mytitlefontsize,
				# yaxis = false, 
	            left_margin = 10mm,
)

# ╔═╡ 4a93db2f-fee0-4402-bb67-bf59144ff457
md"""
### Enrichment Analysis
"""

# ╔═╡ d602140c-c9af-478d-bbba-616e42cccf9f
md"""
#### Over Representation Analysis
"""

# ╔═╡ 525edbf0-3ed1-4def-b01c-f09c9af651d4
md"""
- Computer P-values for each triglycerides with respect to the `age` covariate
"""

# ╔═╡ ec3a528e-ec4b-4149-88ed-de2b58c56f74
begin 
	# check if Age is a predictor
	if ("Age" in vPseudoFrmlNames)
		idx_age = findall(vPseudoFrmlNames.==fix_covar_name("Age"))[1]
		tstat_age = TstatZItg[idx_age, :]
		
		sampleN = size(mX, 1)
		pval_age = ccdf.(TDist(sampleN-1), abs.(TstatZItg[idx_age,:])).*2;

		histogram(
			pval_age, bin= 50, legend = false, grid = false,
			fontfamily = myfont,
			titlefontsize = mytitlefontsize,
			xlabel = "P-value", ylabel = "Counts",
			title = "Histogram of P-values for the Age effect"
		)
	end
end

# ╔═╡ 389cc95f-425c-460b-a4d3-887d10728dd0
function ora_results(predictor_pval, myZgroups, mysubsets, dfMetabolo, dfAnnotation; thresh = 0.05)

    # Initiate results dataframe
    dfORA = DataFrame(
        Subset = mysubsets, 
        Pval_ORA = Vector{Float64}(undef, length(mysubsets))
    );

    # Find index TG with pval less than threshold value , default is 0.05
    idx_sgnf_tg = findall(predictor_pval .< thresh); 
    # Get names of the TGs
    set_sgnf_tg = names(dfMetabolo)[idx_sgnf_tg]

    # Get metbolites population size
    size_metabolites = size(dfMetabolo, 2)
    # Number of success in population
    size_significant = idx_sgnf_tg |> length

    for i in 1:length(mysubsets)
        idx_subset = findall(myZgroups .== mysubsets[i]);
        subset_tg = dfAnnotation.MetaboliteID[idx_subset]

        # Sample size
        size_subset = idx_subset |> length
        # Number of successes in sample
        size_significant_subset = intersect(set_sgnf_tg, subset_tg) |> length

        # Hypergeometric distribution
        # mydist = Hypergeometric(
        #             size_subset, 
        #             size_metabolites-size_subset,
        #             size_significant
        #         )
    
        mydist = Hypergeometric(
            size_significant,        
            size_metabolites - size_significant,
            size_subset
            )
        # Calculate p-value
        dfORA.Pval_ORA[i] = ccdf(mydist, size_significant_subset)
    end

    return dfORA
end;

# ╔═╡ 01b6c77d-205a-4be6-9cde-226d3eb85812
md"""
- Carbon
"""

# ╔═╡ c618b2d2-d57c-48f5-bc61-8cd8620c9f81
begin
	dfORA_tc = ora_results(pval_age, dfRefTG₂.Total_C_cat, lvls_c, dfMetTG[:,2:end], dfRefTG₂)
end

# ╔═╡ 01789761-11a1-452d-a658-cdc6900b15fc
md"""
- Double Bonds
"""

# ╔═╡ 07847e33-6651-48b4-8744-3f512459dee7
begin
	dfORA_db = ora_results(pval_age, dfRefTG₂.Total_DB_cat, lvls_db, dfMetTG[:,2:end], dfRefTG₂)
end

# ╔═╡ 3f0e91f2-dcf2-46cf-a8dd-706076a59835
begin
	dfORA_compare = vcat(dfORA_tc[2:end, :], dfORA_db[2:end, :])

	pval_age_cat_adjusted = ccdf.(TDist(99-1), abs.(TstatZcdbcat[idx_age,:])).*2
	
	dfORA_compare.Pval_MLM_adjusted = pval_age_cat_adjusted[2:end];

	pval_age_db_cat_unadjusted = ccdf.(TDist(sampleN-1), abs.(TstatZdb[idx_age,:])).*2;
	pval_age_tc_cat_unadjusted = ccdf.(TDist(sampleN-1), abs.(TstatZc[idx_age,:])).*2;

	dfORA_compare.Pval_MLM_unadjusted = vcat(
	    pval_age_tc_cat_unadjusted[2:end],
	    pval_age_db_cat_unadjusted[2:end]
    )
	dfORA_compare
end

# ╔═╡ 5d166d4a-e6c9-4bca-a060-32985ca63ac6
begin
if (@isdefined mX)	
	###############################################
	# Carbon chain adjusted by Unstauration level #
	###############################################
	# Generate Z matrix

	dfRefTG₂.Total_C_cont = float.(dfRefTG₂.Total_C)./(maximum(dfRefTG₂.Total_C))#5 # per six unit Carbon
	dfRefTG₂.Total_DB_cont = float.(dfRefTG₂.Total_DB)./(maximum(dfRefTG₂.Total_DB))

	levelsC_DB = ["Intercept", "Double Bond","Carbon Chain", "DoubleBondΞCarbonChain"];

	# Generate Z matrix
	mZcdb_cont = modelmatrix(@formula(y ~ 1 + Total_DB_cont*Total_C_cont).rhs, 
						dfRefTG₂);

	if ("(Intercept)" in vFrmlNames)
		CoefZcdb, CIZcdb, TstatZcdb, varZcdb = getCoefs(
			mYTG, mX, mZcdb_cont; 
			hasXIntercept=false, hasZIntercept= false
		);
	else 
		CoefZcdb, CIZcdb, TstatZcdb, varZcdb = getCoefs(
			mYTG, mX, mZcdb_cont; 
			hasXIntercept=true, hasZIntercept= false
		);
	end
	# CoefZcdb, CIZcdb, TstatZcdb, varZcdb = getCoefs(mYTG, mX, mZcdb_cont);
end
end;	

# ╔═╡ d72b812f-6aa5-4b89-96f1-6941d666c094
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
FreqTables = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MatrixLM = "37290134-6146-11e9-0c71-a5c489be1f53"
MetabolomicsWorkbenchAPI = "19b29032-9db8-4baa-af7b-2b362e62b3d7"
Missings = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.15"
CategoricalArrays = "~0.10.8"
ColorSchemes = "~3.28.0"
DataFrames = "~1.7.0"
DataFramesMeta = "~0.15.4"
Distributions = "~0.25.117"
FreqTables = "~0.4.6"
Images = "~0.26.2"
Latexify = "~0.16.5"
MatrixLM = "~0.2.2"
MetabolomicsWorkbenchAPI = "~0.2.0"
Missings = "~1.2.0"
Plots = "~1.40.7"
PlutoUI = "~0.7.60"
PrettyTables = "~2.4.0"
RecipesBase = "~1.3.4"
Statistics = "~1.11.1"
StatsBase = "~0.34.4"
StatsModels = "~0.7.4"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "defffcd4285fbb75fd4056f3f9be541a26ba4aad"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"
weakdeps = ["JSON", "RecipesBase", "SentinelArrays", "StructTypes"]

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

[[deps.Chain]]
git-tree-sha1 = "9ae9be75ad8ad9d26395bf625dea9beac6d519f1"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.6.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "26ec26c98ae1453c692efded2b17e15125a5bea1"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.28.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "OrderedCollections", "Reexport", "TableMetadataTools"]
git-tree-sha1 = "21a4335f249f8b5f311d00d5e62938b50ccace4e"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.15.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

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
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreqTables]]
deps = ["CategoricalArrays", "Missings", "NamedArrays", "Tables"]
git-tree-sha1 = "4693424929b4ec7ad703d68912a6ad6eff103cfe"
uuid = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
version = "0.4.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

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

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "33485b4e40d1df46c806498c73ea32dc17475c59"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.1"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "33cb509839cc4011beb45bde2316e64344b0f92b"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.9"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "c5c5478ae8d944c63d6de961b19e6d3324812c35"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.0"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fa01c98985be12e5d75301c4527fff2c46fa3e0e"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.1+1"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "b217d9ded4a95052ffc09acc41ab781f7f72c7ba"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.3"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e0884bdf01bbbb111aea77c348368a86fb4b5ab6"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.1"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "a49b96fd4a8d1a9a718dfd9cde34c154fc84fcd5"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "91d501cb908df6f134352ad73cde5efc50138279"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.11"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixLM]]
deps = ["DataFrames", "Distributed", "LinearAlgebra", "Random", "SharedArrays", "Statistics", "StatsModels"]
git-tree-sha1 = "92d6dfd350b05c072e9cc3525c92c837c6d76679"
uuid = "37290134-6146-11e9-0c71-a5c489be1f53"
version = "0.2.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "e9650bea7f91c3397eb9ae6377343963a22bf5b8"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.8.0"

[[deps.MetabolomicsWorkbenchAPI]]
deps = ["CSV", "DataFrames", "HTTP", "JSON3"]
git-tree-sha1 = "16b38874ec66128c3de5a9d08c1fa6157fadb390"
uuid = "19b29032-9db8-4baa-af7b-2b362e62b3d7"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "fe891aea7ccd23897520db7f16931212454e277e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.1"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "58e317b3b956b8aaddfd33ff4c3e33199cd8efce"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ad31332567b189f508a3ea8957a2640b1147ab00"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "f202a1ca4f6e165238d8175df63a7e26a51e04dc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "27f6107dc202e2499f0750c628a848ce5d6e77f5"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

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
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableMetadataTools]]
deps = ["DataAPI", "Dates", "TOML", "Tables", "Unitful"]
git-tree-sha1 = "c0405d3f8189bb9a9755e429c6ea2138fca7e31f"
uuid = "9ce81f87-eacc-4366-bf80-b621a3098ee2"
version = "0.1.0"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "4ab62a49f1d8d9548a1c8d1a75e5f55cf196f64e"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.71"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7b5bbf1efbafb5eca466700949625e07533aff2"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

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
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─1fc288d0-e38d-11ed-1239-49fc6de01db3
# ╟─dd113c37-a08b-4c31-b5bc-7a38d2efdcbe
# ╟─4ffdd121-dceb-4e8a-8f84-f21aad3f84f2
# ╟─e8e4eb9d-12b7-486d-9b40-7174aba0b072
# ╟─da4d2151-6cae-41fc-8966-de03decaf5a8
# ╟─7c18f869-f0ce-44c6-99d3-c0ab2b801001
# ╟─bc027ee4-9912-480a-82ab-7b7f00d29403
# ╟─a77a3328-9fb7-4bfb-b576-00fe281b826e
# ╟─7292bdd6-1e43-464e-829a-013809f06f46
# ╟─af2399d5-6679-4529-a382-b3b0b9601882
# ╟─d5276e0c-8b29-4efa-b1c6-d080658a7a3c
# ╟─edc7f20a-5477-410b-b6d2-64795d0e29dd
# ╟─6d84ebd3-6949-42ff-9b06-f7fd4e7c8f0d
# ╟─5ee27ca3-a0ee-42a5-a0f2-0eddee47f81a
# ╟─adb0c520-3229-48c5-ac4d-5fb2e0bf8093
# ╟─473b48ae-b5c6-4de4-b776-6b664bf46082
# ╟─ef386a1a-f755-49fa-80bb-27d5c338d2e3
# ╟─1cc20658-6725-4a47-a0be-4e42ad2c80b9
# ╟─e41ff6d5-f332-4601-9092-7d182bcd76d2
# ╟─3316f2ff-1ca2-43c3-9c59-dc352b75f9a1
# ╟─60e617ba-4ed4-4353-9afe-61fa677319c0
# ╟─fc95c8db-3373-4665-afa5-e79c65c0d738
# ╟─53fc9db0-28d3-4deb-96f0-91a8e1fe6dbc
# ╟─4d665b9d-a085-482f-b60f-670d3cc2d541
# ╟─1cf4e218-e1e6-47ad-80a2-94fddedecb10
# ╟─7d855e7e-8dff-4de9-b5ee-a7b09d00d978
# ╟─ff672c36-1722-44d6-9003-0a3a57c05bf6
# ╟─b9642a82-4595-4d3b-a240-cc43253fb2a2
# ╟─3f34b8a0-9655-454c-8554-f95821375e90
# ╟─8f39a6ee-fe9b-421e-b4ca-fc88c706f2b9
# ╟─e62d4343-8261-4c49-94ac-06cb9db67f68
# ╟─59c324c4-8b3d-4d0d-a460-13d1a334c8eb
# ╟─ca12f163-518f-4899-8730-0bf5e019f899
# ╟─372f70bf-8b76-4f8a-af3a-955622fd6424
# ╟─143a5bcc-6f88-42f5-b8f9-c8487a6aa030
# ╟─a27ebbdd-0a8b-4f11-a0ab-6a41c991ef88
# ╟─df6118c5-c498-413b-a530-9003fa1340fd
# ╟─8da7da20-885d-43c7-a8d9-1b78baedd11d
# ╟─11d017f1-e15f-47df-9e23-d4d9629ebd63
# ╟─380feaa3-5a52-4803-a561-3f055510ae16
# ╟─5720877a-e485-4a1d-b286-b7572909658c
# ╟─52868e72-74b1-4cfc-9ba6-945fe3c6db8f
# ╟─1a53fd65-8ce5-452f-98a5-129ef8d56329
# ╟─72ab7cdb-f509-48a4-b587-57de058182a8
# ╟─1bb3bb3a-9ffe-4690-817b-ea83ce9e46b6
# ╟─30f08888-0d6c-463d-afce-d60a7deade83
# ╟─825605ec-7c8e-4452-ba4d-63b7d5204d3a
# ╟─d0044ffa-40b7-4675-95fc-e280b28a1207
# ╟─63f2a028-c4a6-4205-bfa5-288a2fcb02c9
# ╟─a094cdf1-bd0c-4c31-ac0b-dd453ca8075f
# ╟─b9b257e4-61bf-4362-bb7f-39931ac0cc0e
# ╟─86b5ad8d-65de-40f8-aff0-b82c1431712d
# ╟─d0969087-77c8-410d-b2c1-6a4a6a6a316d
# ╟─34449f78-6a79-475d-b2cd-ba2beab31ee9
# ╟─47bc75f3-a8fd-4878-9dcd-5ce7151c0820
# ╟─9020e518-ab75-4827-b7b2-d0f25c11a0dc
# ╟─8903ef6a-eff8-4f9c-8f31-374e1e2533c0
# ╟─a4363e40-4420-428f-aec8-9808d39f0e08
# ╟─bb96c16e-2d89-483e-891d-c636f514e30f
# ╟─f578ed91-eec9-4394-8cdc-39f6511a2be9
# ╟─c228eba0-89d6-40fe-8d5e-94a41627c4c9
# ╟─50332a44-81c0-4b80-ab42-517b8196294b
# ╟─63e8f438-abd0-4ee7-8121-cf557e56dc73
# ╟─a599374a-dfd5-45b4-b7bd-9bb728872b55
# ╟─c7fafba4-4c76-4f14-98d6-fd44fe772628
# ╟─ce42d8b6-003e-4e5c-83ef-41372a3526b8
# ╟─87f44b92-74a1-4524-b227-a93a3ac895ad
# ╟─a46aa97d-679a-4b37-a6c0-a6c5148be92f
# ╟─23f880c2-b334-4472-9c93-733ebdc56ecf
# ╟─0f84687b-38b0-4302-99e6-43e3aa2c6443
# ╟─dc3046c8-b3e3-454b-8ea2-4b5a1780db6e
# ╟─0e510137-2641-43a1-af41-fb1fae61ecdf
# ╟─07eb16a0-68bf-4cd5-bd8c-2387497e57d6
# ╟─f77f6c00-3a8a-45f6-a3d0-801693ea6812
# ╟─57ddd596-68b7-43bd-93a8-998d16848f85
# ╟─e80aee70-62ad-446e-884c-c1785218f9cb
# ╟─4c237533-1186-4b7f-81a0-8fe3a271c437
# ╟─7097bd8d-bb77-45fc-a101-c157519abff6
# ╟─836efb58-18ff-4a3e-8a55-5b0a4457a13d
# ╟─4a93db2f-fee0-4402-bb67-bf59144ff457
# ╟─d602140c-c9af-478d-bbba-616e42cccf9f
# ╟─525edbf0-3ed1-4def-b01c-f09c9af651d4
# ╟─ec3a528e-ec4b-4149-88ed-de2b58c56f74
# ╟─389cc95f-425c-460b-a4d3-887d10728dd0
# ╟─01b6c77d-205a-4be6-9cde-226d3eb85812
# ╟─c618b2d2-d57c-48f5-bc61-8cd8620c9f81
# ╟─01789761-11a1-452d-a658-cdc6900b15fc
# ╟─07847e33-6651-48b4-8744-3f512459dee7
# ╟─3f0e91f2-dcf2-46cf-a8dd-706076a59835
# ╟─5d166d4a-e6c9-4bca-a060-32985ca63ac6
# ╟─d72b812f-6aa5-4b89-96f1-6941d666c094
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
