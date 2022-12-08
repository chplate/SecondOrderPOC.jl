using Pkg
Pkg.activate(joinpath(@__DIR__, "results"))
Pkg.add("CSV")
Pkg.add("StatsPlots")
Pkg.add("DataFrames")
Pkg.add("LaTeXStrings")
Pkg.add("Statistics")
Pkg.add("Dates")
using CSV
using StatsPlots
using Plots
using DataFrames
using LaTeXStrings
using Statistics
using Dates
using Colors

# Find duplicates among POC instances
total_counter = 0
counter = 0
elements = Tuple{Int,Int}[]
indices = Int[]
discretizations = Int[]
for j=100:100:500
    for i=5:5:250
        global total_counter +=1
        N = i
        nlin = Int(round(j/N))
        nlin = maximum((1,nlin))
        if !((N,nlin) in elements)
            global counter += 1
            push!(elements, (N,nlin))
            push!(indices, total_counter)
            push!(discretizations, i*nlin)
        end
    end
end
numDistinctInstances = length(elements)


# Set up directories for .csv-files and plots
result_dir = joinpath(dirname(@__FILE__), "results/")
data_dir = joinpath(result_dir, "data/")
plot_dir = joinpath(result_dir, "plots")
for dir in [data_dir, plot_dir]
    if !isdir(dir)
        mkpath(dir)
    end
end

result_filename = joinpath(result_dir, "statistics.txt")
result_file     = open(result_filename, "w")
write(result_file, "POC STO evaluation.\nDate: $(Dates.now())\n")
write(result_file, "POC: Only $numDistinctInstances distinct instances from $total_counter in total.\n")


# Process and gather together all .csv-files in result directory.
write(result_file, "\nPreprocessing raw files:\n")
all_csvs = NamedTuple[]
for (root, dirs, files) in walkdir(result_dir)
    for file in files
        if occursin("results", file)
            write(result_file, joinpath(root, file)*"\n")
            fname = joinpath(root, file)
            df = DataFrame(CSV.File(fname, ignoreemptyrows=true))
            df = df[completecases(df),:]
            bfgs = unique(df.bfgs)
            problem = unique(df.problem)
            if nrow(df)==0
            	continue
         	end
            formulation = df.formulation[1]
            push!(all_csvs, (problem = problem[1], bfgs = bfgs[1],
                                formulation = formulation, df = df,))
        end
    end
end

# Function to remove instances from Dataframes by filtering out the corresponding ID's
remove_duplicates(a) = begin
    _a = tryparse.(Int,last.(split.(a, "_"))) .+ 1
    return _a .∈ [indices]
end


#Walk trough all csv's, ignore instances with BFGS updates, and split data in multiple DataFrames:
#
#- total_df:             includes ALL instances that were computed, even the duplicate instances
#- poc_and_sto_df:       includes instances for POC and STO comparison, POC duplicates are removed
#- poc_then_sto_df:      includes instances in STO formulation after warmstarting with SUR solution with: #switches, order of modes
#- poc_then_stoWS_df:    includes instances in STO formulation after warmstarting with SUR solution with: #switches, order of modes and switching times
#
#The last two DataFrames contain only those instances which were solved successfully in POC formulation.
total_df, poc_df, sto_df, poc_then_sto_df, poc_then_stoWS_df = DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame()
for csv in all_csvs

    df = csv.df
    global total_df = vcat(total_df, df, cols=:union)
    df = subset(df, :bfgs => a -> a .== false)

    if nrow(df) == 0
        continue
    end
    if csv.formulation == "poc"
        _df = subset(df, :ID => a -> remove_duplicates(a))
        _df = subset(_df, :formulation => a -> a .== "poc")
        global poc_df = vcat(poc_df, _df, cols=:union)

        optimalIDs = subset(_df, :status => a -> a.=="LOCALLY_SOLVED").ID
        temp = subset(df, :formulation => a -> a .== "poc_sto")
        temp.N .-= 1 # N then gives number of switches
        temp = subset(temp, :ID => a -> a .∈ [optimalIDs])
        global poc_then_sto_df = vcat(poc_then_sto_df, temp, cols=:union)

        temp = subset(df, :formulation => a -> a.== "poc_stoWS")
        temp = subset(temp, :ID => a -> a .∈ [optimalIDs])
        temp.N .-= 1 #
        global poc_then_stoWS_df = vcat(poc_then_stoWS_df, temp, cols=:union)
    else
        _df = subset(df, :formulation => a -> a.== "sto")
        _df.N .-= 1 # N then gives number of switches
        global sto_df = vcat(sto_df, _df, cols=:union)
    end
end

# Write all relevant .csv-files to disk
CSV.write(joinpath(data_dir, "all_instances.csv"), total_df)
CSV.write(joinpath(data_dir, "poc_df.csv"), poc_df)
CSV.write(joinpath(data_dir, "sto_df.csv"), sto_df)
poc_and_sto_df = vcat(poc_df, sto_df, cols=:union)
CSV.write(joinpath(data_dir, "poc_sto_df.csv"), poc_and_sto_df)
CSV.write(joinpath(data_dir, "poc_then_sto.csv"), poc_then_sto_df)
CSV.write(joinpath(data_dir, "poc_then_stoWS.csv"), poc_then_stoWS_df)

# Write warmstart .csv-files per problem
for problem in unique(poc_then_stoWS_df.problem)
    df_ = subset(poc_then_stoWS_df, :problem => a -> a.==problem)
    CSV.write(joinpath(data_dir, "poc_then_stoWS_$(lowercase(problem)).csv"), df_)
end


write(result_file, "\nDISTRIBUTION OF STATUS, SUCCESS RATES, TIMES:\n")
for (formulation, df) in [("POC", poc_df), ("STO", sto_df), ("STOWS", poc_then_stoWS_df)]
    write(result_file, formulation*":\n")
    for status in unique(df.status)
        if status == "LOCALLY_SOLVED"
            _df = subset(df, :status => a -> a.==status)
            write(result_file, "$formulation: Median $(median(_df.time)) seconds for solving across all problems.\n")
            write(result_file, "$formulation: Mean $(mean(_df.time)) seconds for solving across all problems.\n")
            write(result_file, "$formulation: Standard deviation $(std(_df.time)) seconds for solving across all problems.\n")
            write(result_file, "$formulation: Median $(median(_df.iterations)) iterations for solving across all problems.\n")
            write(result_file, "$formulation: Mean $(mean(_df.iterations)) iterations for solving across all problems.\n")
            write(result_file, "$formulation: Standard deviation $(std(_df.iterations)) iterations for solving across all problems.\n")
            write(result_file, "\n")
        end
    end
    for problem in unique(df.problem)
        _df = subset(df, :problem => a -> a.== problem)
        numSolved = 0
        for status in unique(_df.status)
            _df_stat = subset(_df, :status => a -> a.==status)
            write(result_file,  "$formulation: Problem $problem with status $status: $(nrow(_df_stat)) instances.\n")
            if status == "LOCALLY_SOLVED"
                numSolved = nrow(_df_stat)
                write(result_file,  "$formulation: Problem $problem: Median time for successful instances: $(median(_df_stat.time))\n")
                write(result_file,  "$formulation: Problem $problem: Mean time for successful instances: $(mean(_df_stat.time))\n")
                write(result_file,  "$formulation: Problem $problem: Median iterations for successful instances: $(median(_df_stat.iterations))\n")
                write(result_file,  "$formulation: Problem $problem: Mean iterations for successful instances: $(mean(_df_stat.iterations))\n")
            end
        end
        write(result_file,  "Success rate $formulation for problem $problem : $(numSolved/nrow(_df))\n")
        write(result_file, "\n")
    end
end


# Calculate deviation from minimum for all locally solved instances of POC and STO
objectives_poc_sto_df = DataFrame()
for problem in unique(poc_and_sto_df.problem)
    for formulation in unique(poc_and_sto_df.formulation)
        df_ = subset(poc_and_sto_df, :problem => a -> a.==problem)
        df_ = subset(df_, :formulation => a -> a.==formulation)
        df_ = subset(df_, :status => a -> a.=="LOCALLY_SOLVED")
        min_obj = minimum(df_.objValue)
        df_.min_obj .= min_obj
        df_.deviation_from_best = ((df_.objValue ./ df_.min_obj) .- 1.0) .* 100
        global objectives_poc_sto_df = vcat(objectives_poc_sto_df, df_, cols=:union)
    end
end

# Calculate deviation from minimum for all locally solved instances of STOWS
objectives_stoWS_df = DataFrame()
for problem in unique(poc_then_stoWS_df.problem)
    df_ = subset(poc_then_stoWS_df, :problem => a -> a.==problem)
    df_ = subset(df_, :status => a -> a.=="LOCALLY_SOLVED")
    min_obj = minimum(df_.objValue)
    df_.min_obj .= min_obj
    df_.deviation_from_best = ((df_.objValue ./ df_.min_obj) .- 1.0) .* 100
    global objectives_stoWS_df = vcat(objectives_stoWS_df, df_, cols=:union)
end


write(result_file, "MINIMAL OBJECTIVES FOR LOCALLY SOLVED INSTANCES AND DISTRIBUTION OF DEVIATION:\n")
for (formulation, name, df) in [("poc", "POC", objectives_poc_sto_df), ("sto", "STO", objectives_poc_sto_df), ("poc_stoWS", "STOWS", objectives_stoWS_df)]
    write(result_file, name*":\n")
    _df = subset(df, :formulation => a -> a.==formulation)

    for problem in unique(_df.problem)
        _df_prob = subset(_df, :problem => a -> a.==problem)
        numInstances = nrow(_df_prob)
        write(result_file, "$name: Problem $problem with best found objective $(first(_df_prob.min_obj))\n")
        write(result_file, "$name: Problem $problem with maximal deviation from best found objective in %: $(maximum(_df_prob.deviation_from_best))\n")
        write(result_file, "$name: Problem $problem with median deviation from best found objective in %: $(median(_df_prob.deviation_from_best))\n")
    end
    write(result_file, "\n")
end

close(result_file)

color1 = colorant"#61988E"
color2 = colorant"#E4572E"

# Plot time/iteration for successfully solved instances of POC & STO
poc_and_sto_df.time_pro_iter = poc_and_sto_df.time ./ poc_and_sto_df.iterations
compare_time_per_iter_df = subset(poc_and_sto_df, :status => a -> a .== "LOCALLY_SOLVED")
@df compare_time_per_iter_df groupedboxplot(:problem, :time_pro_iter, color=[color1 color2],
                            xticks=(0.5:1:2.5, [L"\textrm{Egerstedt}", L"\textrm{Lotka}", L"\textrm{Tank}"]),
                            ylabel=L"\textrm{Time \ per \ iteration}", group=:formulation, yscale=:log10,
                            yticks=([10^-2, 10^-1, 10^0, 10^1], [L"\textrm{10^{-2}}" L"\textrm{10^{-1}}" L"\textrm{10^{0}}" L"\textrm{10^{1}}"]),
                            label=[L"\textrm{POC}" L"\textrm{STO}"], linewidth=2, yminorgrid=true,
                            xtickfontsize=14, ytickfontsize=12, yguidefontsize=14)

plot!(dpi=500)
savefig(joinpath(plot_dir, "time_per_iteration.pdf"))
savefig(joinpath(plot_dir, "time_per_iteration"))

# Randomly perturb iterations and N for POC and STO such that in scatter plots points don't overlap
for (i,df) in enumerate([poc_df, sto_df])
    test = Float64[df.iterations df.N]
    final_value = i==1 ? 250 : 50
    for j=1:nrow(df)
        for k=j+1:nrow(df)
            if test[j,:] == test[k,:]
                @debug "Found duplicate at $(j) and  $(k)"
                range_1 = test[k,2] == final_value ? (-.8:.001:0) :  (-1.0:.001:1.0)
                range_2 =  (-2.0:0.001:2.0)
                test[k,:] += [rand(range_1); rand(range_2)]
            end
        end
    end
    df.iterations_perturbed = test[:,1]
    df.N_perturbed = test[:,2]
end
CSV.write(joinpath(data_dir, "scatter_poc.csv"), poc_df)
CSV.write(joinpath(data_dir, "scatter_sto.csv"), sto_df)

for (i,df) in enumerate([poc_df, sto_df])
    ext = i == 1 ? "poc" : "sto"
    for problem in unique(df.problem)
        df_ = subset(df, :problem => a -> a.==problem)
        CSV.write(joinpath(data_dir, "scatter_$(ext)_$(lowercase(problem)).csv"), df_)
    end
end