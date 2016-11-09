##############################################################################
# Type describing one node in the computational tree
# Generates children nodes or terminate
##############################################################################

type Subproblem
  #Define a Subproblem as a remaining lists (boolean) and minimal/most preferred product
  subgraph::BitArray{1} #Subgraph encoded as a list of bits
  minimal_prod::Int64 #Most preferred product in the graph
  name::Int64 #name (TO DO fast hashing)
  depth::Int64 #Depth in the tree, at the moment it is created
  Subproblem() = new(BitArray(1),Int64(0),Int64(0),Int64(0))
  Subproblem(s::BitArray{1},m::Int64,n::Int64,d::Int64) = new(s,m,n,d)
end


function induced_subgraph{T<:SimpleGraph}(g::T, iter)
    n = length(iter)
    isequal(n, length(unique(iter))) || error("Vertices in subgraph list must be unique")
    isequal(n, nv(g)) && return copy(g) # if iter is not a proper subgraph

    h = T(n)
    newvid = Dict{Int, Int}()
    oldvid = Dict{Int, Int}()
    i=1
    for (i,v) in enumerate(iter)
        newvid[v] = i
        oldvid[i] = v
    end

    iterset = Set(iter)
    for s in iter
        for d in out_neighbors(g, s)
            # println("s = $s, d = $d")
            if d in iterset && has_edge(g, s, d)
                newe = Edge(newvid[s], newvid[d])
                add_edge!(h, newe)
            end
        end
    end
    return h,oldvid
end


function connected_components!(label::Vector{Int}, g::SimpleGraph, subgraph::BitArray{1})
    # this version of connected components uses Breadth First Traversal
    # with custom visitor type in order to improve performance.
    # one BFS is performed for each component.
    # This algorithm is linear in the number of edges of the graph
    # each edge is touched once. memory performance is a single allocation.
    # the return type is a vector of labels which can be used directly or
    # passed to components(a)
    nvg = nv(g)
    visitor = LightGraphs.ComponentVisitorVector(label, 0)
    colormap = zeros(Int,nvg)
    que = Vector{Int}()
    sizehint!(que, nvg)
    for v in 1:nvg
        if (subgraph[v]) &(label[v] == 0)
            visitor.labels[v] = v
            visitor.seed = v
            traverse_graph(g, BreadthFirst(), v, visitor, subgraph; colormap=colormap, que=que)
        end
    end
    return label
end


function breadth_first_visit_impl!(
    graph::SimpleGraph,   # the graph
    queue::Vector{Int},                  # an (initialized) queue that stores the active vertices
    colormap::Vector{Int},          # an (initialized) color-map to indicate status of vertices
    visitor::SimpleGraphVisitor,
    subgraph::BitArray{1})  # the visitor

    while !isempty(queue)
        u = shift!(queue)
        open_vertex!(visitor, u)

        for v in out_neighbors(graph, u)
          if subgraph[v]
            v_color::Int = colormap[v]
            # TODO: Incorporate edge colors to BFS
            if !(examine_neighbor!(visitor, u, v, v_color, -1))
                return
            end

            if v_color == 0
                colormap[v] = 1
                discover_vertex!(visitor, v) || return
                push!(queue, v)
            end
          end
        end

        colormap[u] = 2
        close_vertex!(visitor, u)
    end
    nothing
end


function traverse_graph(
    graph::SimpleGraph,
    alg::BreadthFirst,
    s::Int,
    visitor::SimpleGraphVisitor,
    subgraph::BitArray{1};
    colormap = zeros(Int, nv(graph)),
    que = Vector{Int}(),
    )

    colormap[s] = 1
    discover_vertex!(visitor, s) || return
    push!(que, s)

    breadth_first_visit_impl!(graph, que, colormap, visitor,subgraph)
end


function connected_components(g::LightGraphs.Graph,i::BitArray{1})
    label = zeros(Int, nv(g))
    connected_components!(label, g,i)
    c, d = components(label)
    return c
end


function components(labels::Vector{Int})
    d = Dict{Int, Int}()
    c = Vector{Vector{Int}}()
    i = 1
    for (v,l) in enumerate(labels)
      if l > 0
        index = get(d, l, i)
        d[l] = index
        if length(c) >= index
            vec = c[index]
            push!(vec, v)
            c[index] = vec
        else
            push!(c, [v])
            i += 1
        end
      end
    end
    return c, d
end
