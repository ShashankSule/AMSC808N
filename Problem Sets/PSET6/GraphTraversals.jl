module GraphTraversals 

using LightGraphs, MetaGraphs, GraphPlot

#export BFS_Tree, DFS_Tree

"
-------------------------------------------------------------------------------
General utilities
-------------------------------------------------------------------------------
"
function getatts(g::AbstractMetaGraph, attribute::String)
    Atts = []; 
    for i in vertices(g)
        Atts = push!(Atts, get_prop(g, i, Symbol(attribute)));
    end
    
    return Atts 
end

function initialize(mg::AbstractMetaGraph, alg::String) # The initial BFS tree
    BFS_Tree = MetaGraph(0);
    DFS_Tree = MetaGraph(0); 

    if alg == "BFS"
        for i in vertices(mg)
            add_vertex!(BFS_Tree);
            set_props!(BFS_Tree, i, Dict(:color => "WHITE", :dist => Inf, :parent => i))
        end
        return BFS_Tree

    elseif alg == "DFS"
        for i in vertices(mg)
            add_vertex!(DFS_Tree);
            set_props!(DFS_Tree, i, Dict(:color => "WHITE", 
                                         :d => Inf, 
                                         :parent => i, :f => Inf));
        end
        return DFS_Tree
    else 
        print("You in the wrong town, pal")
    end

end

"
--------------------------------------------------------------------------
BFS subroutines 
--------------------------------------------------------------------------
"
function bfs_connected(mg::AbstractMetaGraph, BFS_Tree::AbstractMetaGraph, seed::Int64)
    
    set_props!(BFS_Tree, seed, Dict(:color => "GREY", :dist => 0.0, :parent => seed));
    FOFQ = Int64[]; 
    FOFQ = push!(FOFQ,seed); # First in First Out Queue

    #Connected Graph subroutine 
    while length(FOFQ) != 0 
        current = FOFQ[1]; #Current vertex
        #print("\nCurrently at ", current); 

        # Visit subroutine 
        current_distance = get_prop(BFS_Tree, current, :dist);
        for i in neighbors(mg, current)

            if get_prop(BFS_Tree, i, :color) == "WHITE"
                set_props!(BFS_Tree, i, Dict(:color => "GREY", 
                                             :dist => current_distance + 1.0,
                                             :parent => current)); 
                add_edge!(BFS_Tree, current, i);
                push!(FOFQ, i); 
            end

        end

        # Recolour current vertex and take it out of Q 
        set_prop!(BFS_Tree, current, :color, "BLACK"); 
        deleteat!(FOFQ,1); 
        print("\rFinished Exploring ", current, ".Queue has ", length(FOFQ)," members remaining"); 
    end

    return BFS_Tree
end

function my_bfs(mg::AbstractMetaGraph,seed::Int64)
    
    "
    Computes a BFS tree for given graph g. Inputs:
    
    g--MetaGraph on which BFS tree is to be computed 
    seed--initial vertex 
    "
    
    BFS_Tree = initialize(mg, "BFS");
    iter = 0; 
    c_sizes = []; 
    while sum(Int64.(getatts(BFS_Tree, "color") .== "WHITE")) != 0
        iter = iter+1;
        #print("\n Computing the ",iter,"th component..."); 
        
        if iter > 1 
            seed = findfirst(getatts(BFS_Tree, "color") .== "BLACK"); 
        end
        
        BFS_Tree = bfs_connected(mg, BFS_Tree, seed); 
        c_sizes = push!(c_sizes, sum(Int64.(getatts(BFS_Tree, "color") .== "BLACK")))
    end
    
    sizes = []; 
    push!(sizes, c_sizes[1]); 
    if iter > 1 
        for i in 2:iter 
            sizes = push!(sizes, c_sizes[i] - c_sizes[i-1]);
        end
    end
    
    set_prop!(BFS_Tree, :ncc, iter)
    set_prop!(BFS_Tree, :cc_sizes, Float64.(sizes))

    return BFS_Tree
end

"
---------------------------------------------------------------------------------
DFS subroutines
--------------------------------------------------------------------------------
"

function DFS_VISIT(mg::AbstractMetaGraph, DFS_Tree:: AbstractMetaGraph, u::Int64, time::Int64)
    "
    Inputs:
    mg--graph that we traverse 
    u--active vertex 
    time--time at which we enter 
    "
    
    time = time + 1; 
    set_prop!(DFS_Tree, u, :d, time); 
    set_prop!(DFS_Tree, u, :color, "GREY"); 
    
    for v in neighbors(mg, u)
        if get_prop(DFS_Tree, v, :color) == "WHITE"
            set_prop!(DFS_Tree, v, :parent, u);
            add_edge!(DFS_Tree, v, u); 
            time, DFS_Tree = DFS_VISIT(mg,DFS_Tree, v,time); 
        end
    end
    
    set_prop!(DFS_Tree, u, :color, "BLACK"); 
    time = time + 1;
    set_prop!(DFS_Tree, u, :f, time);
    return time, DFS_Tree
end

function dfs_connected(mg::AbstractMetaGraph, DFS_Tree::AbstractMetaGraph)
    time = 0; 
    c_sizes = []; 
    #iter = 1; 
    for u in vertices(DFS_Tree)
        if get_prop(DFS_Tree, u, :color) == "WHITE"
            #print("\nComputing component #",iter)
            time, DFS_Tree = DFS_VISIT(mg, DFS_Tree, u, time); 
            c_sizes = push!(c_sizes, sum(Int64.(getatts(DFS_Tree, "color") .== "BLACK"))); 
            #iter = iter+1;
        end
    end
    
    sizes = [c_sizes[1]]; 
    
    for i = 2:length(c_sizes)
        sizes = push!(sizes, c_sizes[i] - c_sizes[i-1]);
    end
    
    set_prop!(DFS_Tree, :cc, length(sizes)); 
    set_prop!(DFS_Tree, :cc_sizes, sizes); 

    return DFS_Tree
end

function my_dfs(mg::AbstractMetaGraph)
    DFS_Tree = initialize(mg,"DFS")
    
    return dfs_connected(mg, DFS_Tree)
end

end