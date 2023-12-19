### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 611fd360-9c4b-11ee-2b7b-330fb8968bdb

##### I would like to thank Diderot the author and developer of the following https://github.com/rschwarz/Diderot.jl, I used his code as my primary implementation of decision diagrams (DDs) when I started learning about DDs. Moreover, I used his code as a starting point for my implementation of the idea in the paper.   

module DD

using DataStructures
using AutoHashEquals
using DelimitedFiles
using Clustering
using Random


# A structure to implementation an "arc" in DD; an arc has three parts (i.e. tail is the state from which the arc comes out, decision which is to store the decision on variables, value which is associated with the cost function in DD). 
@auto_hash_equals struct Arc{S,D,V}
    tail::S
	decision::D
    value::V
	
end

# There is a node associated to every state S in a DD which stores corresponding objective value for and incoming arc to state S.
@auto_hash_equals struct Node{S,D,V}
    obj::V
	inarc::Union{Arc{S,D,V},Nothing}
	
	function Node{S,D,V}(obj, inarc=nothing) where {S,D,V} 
        new(obj, inarc) 
    end
end

# A layer in DD is a set of states (nodes) corresponding to a decision variable.
@auto_hash_equals struct Layer{S,D,V}
    nodes::Dict{S,Node{S,D,V}}
	
    function Layer{S,D,V}(nodes=Dict()) where {S,D,V}
        return new(nodes)
    end
end
function Base.iterate(layer::Layer{S,D,V}) where {S,D,V}
    return iterate(layer.nodes)
end

function Base.iterate(layer::Layer{S,D,V}, state) where {S,D,V}
    return iterate(layer.nodes, state)
end

function Base.length(layer::Layer{S,D,V}) where {S,D,V}
    return length(layer.nodes)
end

function Base.haskey(layer::Layer{S,D,V}, state::S) where {S,D,V}
    return haskey(layer.nodes, state)
end

function Base.getindex(layer::Layer{S,D,V}, state::S) where {S,D,V}
    return getindex(layer.nodes, state)
end

function Base.setindex!(
    layer::Layer{S,D,V}, node::Node{S,D,V}, state::S
) where {S,D,V}
    return setindex!(layer.nodes, node, state)
end


@auto_hash_equals struct Diagram{S,D,V}
    layers::Vector{Layer{S,D,V}}
end

function Diagram(initial::Layer{S,D,V}) where {S,D,V}
    return Diagram{S,D,V}([initial])
end

function Diagram(instance)
    state = initial_state(instance)
    S = typeof(state)
    D = domain_type(instance)
    V = value_type(instance)
    node = Node{S,D,V}(zero(V))
    root = Layer{S,D,V}(Dict(state => node))
    return Diagram(root)
end


######  interfaces

function initial_state() end
function domain_type end
function value_type end
function transitions end


######   generic functions

# This function is an implementation of the lines 2 to 5 of the algorithm in the paper all in one place (i.e. it gets a variable and builds its corresponding layer).
function build_layer(instance, diagram::Diagram{S,D,V}, variable, problem_type) where {S,D,V}
    layer = Layer{S,D,V}()

    # Collect new states
	for (state, node) in diagram.layers[end] 
		for (arc, new_state) in transitions(instance, state, variable)
			if !haskey(layer, new_state) 
				layer[new_state] = Node{S,D,V}(node.obj + arc.value, arc)
			else
				if problem_type == "minimization"
					if layer[new_state].obj > node.obj + arc.value 
						layer[new_state] = Node{S,D,V}(node.obj+arc.value, arc)
					end
				elseif problem_type == "maximization"
					if layer[new_state].obj < node.obj + arc.value 
						layer[new_state] = Node{S,D,V}(node.obj + arc.value, arc)
					end	
				end
			end
		end
    end
    return layer
end




# This is the main function (i.e. the implementation of the whole algorithm in the paper to build an approximate DD). The arguments' descriptions are as followes:

# W: Given maximum width,

# problem_type: is the problem a minimization or maximization problem,

# bound_type: is a restricted (provides a primal bound) or a relaxed (provides a dual bound) DD going to be compiled,

# selection_heuristic: should "sortObj" heuristic be used for node selection or "clustering-based" heuristic,

# k: number of clusters when KMeans cluster is being used (argument is active only if "clustering-based" heuristic is chosen),	

# max_iter: maximum iteration of KMeans clustering algorithm (argument is active only if "clustering-based" heuristic is chosen).	

	
function approximate_DD!(
    diagram::Diagram{S,D,V}, instance; W::Int64, problem_type::AbstractString, bound_type::AbstractString, selection_heuristic::AbstractString, seed::Int64, clustering::AbstractString, clusters::Int64, max_iter_cluster::Int64) where {S,D,V}
    
	@assert clusters <= W
	@assert (problem_type == "minimization" || problem_type == "maximization")
	@assert (bound_type == "primal" || bound_type == "dual")
	@assert (selection_heuristic == "sortObj" || selection_heuristic == "clustering-based" || selection_heuristic == "random")
	@assert (clustering == "kmeans" || clustering == "kmedoids")
	@assert length(diagram.layers) == 1  

	instance_size = length(instance)
	
	for variable in 1:instance_size
        
		layer = build_layer(instance, diagram, variable, problem_type)
		
		if length(layer) > W 
			
			if selection_heuristic == "sortObj"
			
				layer = sortObj(layer, W, problem_type, bound_type)
			
			elseif selection_heuristic == "random"
				
				layer = random(layer, W, problem_type, bound_type, seed)
				
			elseif selection_heuristic == "clustering-based"
				
				layer = clustering_based(layer, problem_type, bound_type, clusters, max_iter_cluster, clustering)
				
			end
		end
		push!(diagram.layers, layer)
	end	
	
	# Terminal node (last layer relaxed into a terminal node)
	diagram.layers[end] = last_layer_into_terminal(diagram.layers[end], problem_type) 
	terminal_node = only(values(diagram.layers[end].nodes))
	return terminal_node.obj
end


# This function will become activated if one desires to use sortObj node selection heuristic when building a restricted/relaxed DD (when building an approximate DD).

function sortObj(layer::Layer{S,D,V}, W, problem_type, bound_type) where {S,D,V}
	
	decreasing_order = true
	if problem_type == "minimization"
		decreasing_order = false
	end
	
	# Sort states according to their associated objective value 
	sorted = sort!(collect(layer), by=tup -> tup.second.obj, rev=decreasing_order)
	
	if bound_type == "primal"
		
		# for restricting the layer, we select the best W (maximum width) states 
		return Layer{S,D,V}(Dict(sorted[1:W]))
	
	elseif bound_type == "dual"
		
		# for relaxing a layer, we first select the best W-1 (maximum width - 1) to kepp in the layer, then merge the rest of them (*** merge) 
		new_layer = Layer{S,D,V}(Dict(sorted[1:(W - 1)]))
		state_dimension = length(sorted[1].first)
		states = @view sorted[W:end]

		tuple = Tuple(Tuple(states[i].first[j] for i in 1:length(states)) for j in 1:state_dimension)
		
		# (*** merge), in all 5 problems we considered the merge operation is to take the element-wise minimum across all the states that are going to be merged
		merged_state = Tuple(minimum(tuple[i]) for i in 1:state_dimension)
		
		merged_node = Node{S,D,V}(states[1].second.obj, states[1].second.inarc) 
		
		if !haskey(new_layer, merged_state)
			new_layer[merged_state] = merged_node
		else
			# if the merged node already exists in the layer, then take the dominant one
			if problem_type == "minimization"
				
				if new_layer[merged_state].obj > merged_node.obj
					new_layer[merged_state] = merged_node
				end
			
			elseif problem_type == "maximization"
				
				if new_layer[merged_state].obj < merged_node.obj
					new_layer[merged_state] = merged_node
				end				
			end
		end
		
		return new_layer
		
	end
	
end


# This function will become activated if one desires to use random node selection heuristic with a given seed when building a restricted/relaxed DD (when building an approximate DD).

function random(layer::Layer{S,D,V}, W, problem_type, bound_type, random_seed) where {S,D,V}
	
	decreasing_order = true
	if problem_type == "minimization"
		decreasing_order = false
	end
	
	# assign numbers to states 
	collection = shuffle!(MersenneTwister(random_seed), collect(layer))
	
	if bound_type == "primal"
		
		# for restricting the layer, we select W (maximum width) states randomly
		return Layer{S,D,V}(Dict(collection[1:W]))
	
	elseif bound_type == "dual"
		
		# for relaxing a layer, we first select W-1 (maximum width - 1) states randomly to keep in the layer, then merge the rest of them (*** merge) 
		new_layer = Layer{S,D,V}(Dict(collection[1:(W - 1)]))
		state_dimension = length(collection[1].first)
		helper = @view collection[W:end]
		states = sort!(helper, by=tup -> tup.second.obj, rev=decreasing_order)

		tuple = Tuple(Tuple(states[i].first[j] for i in 1:length(states)) for j in 1:state_dimension)
		
		# (*** merge), in all 5 problems we considered the merge operation is to take the element-wise minimum across all the states that are going to be merged
		merged_state = Tuple(minimum(tuple[i]) for i in 1:state_dimension)
		
		merged_node = Node{S,D,V}(states[1].second.obj, states[1].second.inarc) 
		
		if !haskey(new_layer, merged_state)
			new_layer[merged_state] = merged_node
		else
			if problem_type == "minimization"
				
				if new_layer[merged_state].obj > merged_node.obj
					new_layer[merged_state] = merged_node
				end
			
			elseif problem_type == "maximization"
				
				if new_layer[merged_state].obj < merged_node.obj
					new_layer[merged_state] = merged_node
				end				
			end
		end
		
		return new_layer
		
	end
	
end


# This function will become activated if one desires to use clustering-based node selection heuristic when building a restricted/relaxed DD (when building an approximate DD)

function clustering_based(layer::Layer{S,D,V}, problem_type, bound_type, n_clusters, max_iter_cluster, cluster) where {S,D,V}
	
	new_layer = Layer{S,D,V}()
	decreasing_order = true 		
	if problem_type == "minimization"
		decreasing_order = false
	end
	
	# cluster the states of a layer that is going to be restricted/relaxed 
	clusters = cluster_states(layer, n_clusters, max_iter_cluster, cluster)
	
	if bound_type == "primal"
		
		# in case of a restricted DD select and keep the best state from each cluster 		  in the layer
		for key in keys(clusters)
			states = sort!(collect(clusters[key]), by= tup -> layer[tup].obj, rev=decreasing_order)
			new_layer[states[1]] = layer[states[1]]
		end
		
	elseif bound_type == "dual"

		# in case of a relaxed DD merge the states of every cluster into a merged node
		for key in keys(clusters)
			
			states = sort!(collect(clusters[key]), by= tup -> layer.nodes[tup].obj, rev=decreasing_order)
			
			state_dimension = length(states[1])

			tuple = Tuple(Tuple(states[i][j] for i in 1:length(states)) for j in 1:state_dimension)

			merged_state = Tuple(minimum(tuple[i]) for i in 1:state_dimension)

			merged_node = Node{S,D,V}(layer[states[1]].obj, layer[states[1]].inarc)
			
			if !haskey(new_layer, merged_state)
				new_layer[merged_state] = merged_node
			else
				if problem_type == "minimization"
					if new_layer[merged_state].obj > merged_node.obj
						new_layer[merged_state] = merged_node
					end

				elseif problem_type == "maximization"

					if new_layer[merged_state].obj < merged_node.obj
						new_layer[merged_state] = merged_node
					end				

				end
			end

		end		
	end
	
	return new_layer
	
end

# This funtion takes the last layer and merge all its states into a single merged state which is the terminal 
function last_layer_into_terminal(layer::Layer{S,D,V}, problem_type) where {S,D,V}
	
	decreasing_order = true
	if problem_type == "minimization"
		decreasing_order = false
	end

	states = sort!(collect(layer), by=tup -> tup.second.obj, rev=decreasing_order)
	
	state_dimension = length(states[1].first) 

	tuple = Tuple(Tuple(states[i].first[j] for i in 1:length(states)) for j in 1:state_dimension)
		
	merged_state = Tuple(minimum(tuple[i]) for i in 1:state_dimension)
		
	return Layer{S,D,V}(Dict([merged_state => Node{S,D,V}(states[1].second.obj, states[1].second.inarc)]))
end


# This function clusters the states of a given layer using the built-in KMeans clustering function in Julia.

function cluster_states(layer::Layer{S,D,V}, n_clusters::Int64, max_iterations::Int64, clustering_approach) where {S,D,V}
    
	states = collect(keys(layer.nodes))
	n_datapoints = length(states)
	feature_dimension = length(states[1])
	feature_vectors = Matrix{Int64}(undef, feature_dimension, n_datapoints)
	for i in 1:n_datapoints
		for j in 1:feature_dimension
			feature_vectors[j,i] = states[i][j]
		end
	end

	##  in case objective value is needed to have a coordinate in feature vector (e.g. in MKP we used it) comment the previous 6 lines and instead uncomment the following 7 lines
	
	# feature_vectors = Matrix{Int64}(undef, feature_dimension + 1, n_datapoints)
	# for i in 1:n_datapoints
	# 	for j in 1:feature_dimension
	# 		feature_vectors[j,i] = states[i][j]
	# 	end
	# 	feature_vectors[feature_dimension+1, i] = layer.nodes[states[i]].obj
	# end	
	
	clustering = nothing	
	if clustering_approach == "kmeans"
		clustering = kmeans(feature_vectors, n_clusters; maxiter = max_iterations)
	elseif clustering_approach == "kmedoids"
		# distance = pairwise(Euclidean(), feature_vectors, dims=2)
		distance = pairwise(SqEuclidean(), feature_vectors, dims=2)
		clustering = kmedoids(distance, n_clusters; maxiter = max_iterations)
	end

    clusters = Dict{Int64, Set{S}}()
	for i in 1:n_datapoints
		if !haskey(clusters, clustering.assignments[i])
			clusters[clustering.assignments[i]] = Set{S}()
		end
		push!(clusters[clustering.assignments[i]], states[i])
	end
	println(clusters)
	@assert length(clusters) <= n_clusters
	@assert length(layer) == sum([length(clusters[i]) for i in keys(clusters)])
	return clusters
end


##############################################################################
#######  problem specific part
# In this part specification of every problem (i.e. format of an input (instance) of the problem, variable domain, initial state (root state), transition function) is written.

##########   #  0/1 Knapsack 

module KP

using ..DD
using ..DD: Arc, Node, Layer

struct Instance
    profit::Vector{Int64}
    weight::Vector{Int64}
    capacity::Int64

    function Instance(profit, weight, capacity)
		@assert length(profit) == length(weight)
        new(profit, weight, capacity)
    end
end

Base.length(instance::Instance) = length(instance.profit) 
DD.domain_type(instance::Instance) = Int64
DD.value_type(instance::Instance) = Int64
DD.initial_state(instance::Instance) = Tuple(0)

function DD.transitions(instance::Instance, state, variable)
	
	results = Dict{Arc{typeof(state), Int64, Int64}, typeof(state)}()	
	new_state = Tuple(instance.weight[variable] + state[1])
	
	# take/add the item
	if new_state[1] <= instance.capacity 
		arc = Arc(state, 1, instance.profit[variable])
		results[arc] = new_state
	end

    # don't take the item
    results[Arc(state, 0, 0)] = state 
	
    return results
end

end # module


##############################################################################
##########   #  MultiDimensional Knapsack

module MKP

using ..DD
using ..DD: Arc, Node, Layer

struct Instance
    profit::Vector{Int64}
    weight::Vector{Vector{Int64}}
    capacity::Vector{Int64}

    function Instance(profit, weight, capacity)
        @assert length(capacity) == length(weight[1])
		@assert length(profit) == length(weight)
        new(profit, weight, capacity)
    end
end

Base.length(instance::Instance) = length(instance.profit) 
DD.domain_type(instance::Instance) = Int64
DD.value_type(instance::Instance) = Int64
DD.initial_state(instance::Instance) = Tuple(0 for _ in 1:length(instance.weight[1]))

function DD.transitions(instance::Instance, state, variable)
	
	results = Dict{Arc{typeof(state), Int64, Int64}, typeof(state)}()	
	new_state = Tuple(instance.weight[variable][i] + state[i] for i in 1:length(state))
			
	# take the item
	if all(new_state .<= instance.capacity) 
		arc = Arc(state, 1, instance.profit[variable])
		results[arc] = new_state
	end
	
    # don't take the item
    results[Arc(state, 0, 0)] = state 
	
    return results
end

end # module


##############################################################################
##########    # Weighted Number of Tardy Jobs on a Single Machine

module WNTJSM

using ..DD
using ..DD: Arc, Node, Layer

struct Instance
	processing_time::Vector{Int64}
	weight::Vector{Int64}
	due_date::Vector{Int64}

    function Instance(processing_time, weight, due_date)
        new(processing_time, weight, due_date)
    end
end

Base.length(instance::Instance) = length(instance.processing_time) 
DD.domain_type(instance::Instance) = Bool
DD.value_type(instance::Instance) = Int64
DD.initial_state(instance::Instance) = Tuple(0) 

function DD.transitions(instance::Instance, state, variable)
    
	results = Dict{Arc{typeof(state), Bool, Int64}, typeof(state)}()
	process_time = instance.processing_time[variable]
	weight = instance.weight[variable]
	due = instance.due_date[variable]
	
	# ealy job 
	if state[1] + process_time <= due
		arc = Arc(state, true, 0)
		results[arc] = Tuple(process_time + state[1])
	end
	
    # tardy job
	arc = Arc(state, false, weight)
	results[arc] = state

	return results
end

end # module
	
##############################################################################	
##########    #  Sum of Cubed Job Completion Times on Two Identical Machines

module SCJCTTIM

using ..DD
using ..DD: Arc, Node, Layer

struct Instance
	processing::Vector{Int64}

    function Instance(processing)
        new(processing)
    end
end

Base.length(instance::Instance) = length(instance.processing) 
DD.domain_type(instance::Instance) = Int64
DD.value_type(instance::Instance) = Int64
DD.initial_state(instance::Instance) = (0,0) 


function DD.transitions(instance::Instance, state, variable)
    
	results = Dict{Arc{typeof(state), Int64, Int64}, typeof(state)}()
	machine_1 = state[1]
	machine_2 = state[2]
	process_time = instance.processing[variable]

	# Assign the job to Machine1 
	arc = Arc(state, 1, (process_time + machine_1)^3)
	results[arc] = (process_time + machine_1, machine_2)
	
    # Assign the job to Machine2
	arc = Arc(state, 2, (process_time + machine_2)^3)
	results[arc] = (machine_1, machine_2 + process_time)

	return results
end

end # module


##############################################################################
##########   #  Total Weighted Job Completion Time on Two Identical Machines

module TWJCTTIM

using ..DD
using ..DD: Arc, Node, Layer

struct Instance
	processing_time::Vector{Int64}
	weight::Vector{Int64}

    function Instance(processing_time, weight)
        new(processing_time, weight)
    end
end

Base.length(instance::Instance) = length(instance.processing_time) 
DD.domain_type(instance::Instance) = Int64
DD.value_type(instance::Instance) = Int64
DD.initial_state(instance::Instance) = (0,0) 


function DD.transitions(instance::Instance, state, variable)
    
	results = Dict{Arc{typeof(state), Int64, Int64}, typeof(state)}()
	machine_1 = state[1]
	machine_2 = state[2]
	process_time = instance.processing_time[variable]
	weight = instance.weight[variable]

	# Assign the job to Machine1 
	arc = Arc(state, 1, (process_time + machine_1) * weight)
	results[arc] = (process_time + machine_1, machine_2)
	
    # Assign the job to Machine2
	arc = Arc(state, 2, (process_time + machine_2) * weight)
	results[arc] = (machine_1, machine_2 + process_time)

	return results
end

end # module

##########################  Reading functions   ##############################  
# The following functions are written to read instances of problems from instance files and then create an instance of the problem.


####   NOTE
# Please adjust the file address in the following functions according to the location of the folder "instances" on your device.   #### 


function read_KP(f::Int64)
	
	data = readdlm("address on your devide\\instances\\KP_instances\\KP_$f.txt")	
	n_items = data[end,1]
	capacity = data[end,2]
	profit = Base.zeros(Int64, n_items)
	weight = Base.zeros(Int64, n_items)
	for i in 1:n_items
		profit[i] = data[1,i]
		weight[i] = data[2,i]
	end	

	return DD.KP.Instance(profit, weight, capacity)
end


function read_MKP(f::Int64)
	
	data = readdlm("address on your device\\instances\\MKP_instances\\MKP_$f.txt")
	items = data[end,1]
	constraints = data[end,2]
	profit = Base.zeros(Int64, items)
	weight = Vector{Vector{Int}}(undef, items)
	capacities = Base.zeros(Int, constraints)
	for i in 1:items
		weight[i] = Base.zeros(Int, constraints)
		for k in 1:constraints
			weight[i][k] = data[k,i]
		end
	end
	for i in 1:items
		profit[i] = Int64(data[end-2,i])
	end
	for i in 1:constraints
		capacities[i] = data[end-1,i]
	end	

	return DD.MKP.Instance(profit, weight, capacities)
end


function read_scheduling(f::Int64, problem::AbstractString)
	
	@assert (problem == "WNTJSM" || problem == "TWJCTTIM" || problem == "SCJCTTIM")
	
	data = []
	
	if problem == "TWJCTTIM" || problem == "SCJCTTIM"
	
		data = readdlm("address on your devide\\instances\\Scheduling_instances\\SCJCTTIM_TWJCTTIM\\instance_$f.txt")
	
	elseif problem == "WNTJSM"
		
		data = readdlm("address on your devide\\instances\\Scheduling_instances\\WNTJSM\\500wt_$f.txt")
	
	end
		
	n_jobs = data[1,1]
	
	processings = zeros(Int64, n_jobs)
	weights = zeros(Int64, n_jobs)
	dues = zeros(Int64, n_jobs)
	
	for i in 1:n_jobs
		processings[i] = data[2,i]
		weights[i] = data[3,i]
		dues[i] = data[4,i]
	end
	
	if problem == "SCJCTTIM"
		return DD.SCJCTTIM.Instance(sort!(processings, by=tup->tup, rev=false))
	elseif problem == "TWJCTTIM"
		return DD.TWJCTTIM.Instance(sort!(processings, by=tup->processings[tup]/weights[tup], rev=false), sort!(weights, by=tup->processings[tup]/weights[tup], rev=false))
	elseif problem == "WNTJSM"
		return DD.WNTJSM.Instance(sort!(processings, by=tup->dues[tup], rev=false), sort!(weights, by=tup->dues[tup], rev=false), sort!(dues, by=tup->tup, rev=false))
	end
end


end



# ╔═╡ 2ba743bf-3f8d-4313-82bb-094e74a50894
######################     setting parameters    ######################

comment this line, then run the cell

begin
	
	### desired maximum width
	max_width = 10 
	
	### desired bound type, i.e. primal or dual
	bound = "primal" 
	
	### desired selection heuristic, i.e. sortObj, clustering-based, random
	Heuristic = "sortObj" 
	
	### desired clustering approach, i.e. kmeans, kmedoids
	clustering_approach = "kmeans" 
	
	### max iteration for clustering algorithm
	cluster_iteration = 50 
	
	### desired number of clusters, must be smaller than maximum width
	n_clusters = 1 
	
	### desired seed for random selection
	random_seed = 1234 

end

# ╔═╡ f82ba65c-7940-4029-b799-35217ebdcb29

######################     Run cell for 0/1 Knapsack problem    ######################

comment this line, then run the cell

begin 
	
	### creating an instance
	problem_instance = DD.read_KP(i)
	
	### creating a diagram containing only the root node
	problem_diagram = DD.Diagram(problem_instance)

	### buiding the BDD
	DD.approximate_DD!(problem_diagram, problem_instance; W=max_width, problem_type="maximization", bound_type=bound, selection_heuristic=Heuristic, seed=random_seed, clustering=clustering_approach, clusters=n_clusters, max_iter_cluster=cluster_iteration)

	
end	

# ╔═╡ d92996fc-72dd-476f-81aa-f27ec43529a6

###############     Run cell for Multidimentional Knapsack problem    ##############

comment this line, then run the cell

begin 
	
	### creating an instance
	problem_instance = DD.read_MKP(i)
	
	### creating a diagram containing only the root node
	problem_diagram = DD.Diagram(problem_instance)

	### buiding the BDD
	DD.approximate_DD!(problem_diagram, problem_instance; W=max_width, problem_type="maximization", bound_type=bound, selection_heuristic=Heuristic, seed=random_seed, clustering=clustering_approach, clusters=n_clusters, max_iter_cluster=cluster_iteration)

	
end	

# ╔═╡ e6589f9a-f78d-4eb1-b4c0-e0331378695f

######################    Run cell for Scheduling problems     ######################

comment this line, then run the cell

begin                           
	
	### desired problem
	problem = "WNTJSM"
	# problem = "TWJCTTIM" 
	# problem = "SCJCTTIM"

	### creating an instance
	problem_instance = DD.read_scheduling(i, problem)
	
	### creating a diagram containing only the root node
	problem_diagram = DD.Diagram(problem_instance)

	### buiding the BDD
	DD.approximate_DD!(problem_diagram, problem_instance; W=max_width, problem_type="minimization", bound_type=bound, selection_heuristic=Heuristic, seed=random_seed, clustering=clustering_approach, clusters=n_clusters, max_iter_cluster=cluster_iteration)

end	

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AutoHashEquals = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
AutoHashEquals = "~1.0.0"
Clustering = "~0.15.5"
DataStructures = "~0.18.15"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AutoHashEquals]]
deps = ["Pkg"]
git-tree-sha1 = "7fc4d1532a3df01af51bae5c1d20389f5aeea086"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"

[[ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "05f9816a77231b07e634ab8715ba50e5249d6f76"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.5"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"

[[DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

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

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "3ef8ff4f011295fd938a521cb605099cecf084ca"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.15"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2aded4182a14b19e9b62b063c0ab561809b5af2c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.8.0"

[[StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═611fd360-9c4b-11ee-2b7b-330fb8968bdb
# ╠═2ba743bf-3f8d-4313-82bb-094e74a50894
# ╠═f82ba65c-7940-4029-b799-35217ebdcb29
# ╠═d92996fc-72dd-476f-81aa-f27ec43529a6
# ╠═e6589f9a-f78d-4eb1-b4c0-e0331378695f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
