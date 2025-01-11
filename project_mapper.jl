import Clustering
import Distances
import LinearAlgebra: norm, det, inv
import Statistics: cov, mean, std
import MultivariateStats
import PlyIO
import MeshIO
import GeometryBasics
import FileIO
import SparseArrays
# we need a function to read the .ply files

using FileIO

function get_abstract_matrix(path::String)::Matrix{Float64}
    data = PlyIO.load_ply(path)  # Efficient `.ply` parser
    return hcat(data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"])
end

airplane = get_abstract_matrix("data/non-articulated/airplanes/airplanesPly/b16.ply")

# we create a structure for Mapper representation, we have adjacency matrix, filter range, the constructed patches and their centers

struct Mapper
    adjacency_matrix::AbstractMatrix{<:Integer}
    filter_range::Vector{<:Real}
    found_patches::Vector{Vector{<:Integer}}
    centers_of_patches::Matrix{<:Real}
end

# helping debugging functions

function point_distances2(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; p=2.0)
    return norm(x - y, p)
end

function filter_general_debug(X::AbstractMatrix{<:Real}; kwargs...)
    ϵ = 1.0
    for (a, b) in kwargs
        if a == :ϵ
            ϵ = b
        end
    end

    # Kernel matrix with dimensions (n_rows, n_rows)
    pairwise_kernel_matrix = MultivariateStats.pairwise((a, b) -> exp(-(1/ϵ) * norm(a - b)^2), X)

    # Sum over rows
    sum_over_matrix = sum(pairwise_kernel_matrix, dims=2)

    # Normalize to get filter values corresponding to rows of X
    normalized = vec(sum_over_matrix ./ sum(sum_over_matrix))

    return normalized[1:size(X, 1)]  # Match filter size to data rows
end
# Cluster measurement functions

"""
first measurement function is going to be the distance of a point from other points,
    we basically calculate p-norm and create distance matrix
"""

function point_distances(X::AbstractMatrix{<:Real}; kwargs...)
    p = 2.0
    for (a, b) in kwargs
        if a == :p
            p = b
        end
    end
    N = size(X, 2)
    D_matrix = MultivariateStats.pairwise((x, y) -> LinearAlgebra.norm(x - y, p), X)
    vec(sum(D_matrix, dims=2) ./ N) .^ (1 / p)
end

"""
second measurement function is based on the z-coordinate
"""
function z_coordinate(X::AbstractMatrix{<:Real}; kwargs...)
    return X[:, 3]
end

# Filtering functions

"""
2 filter functions, first one useful for gaussian clusters to highlight core cluster regions
and second one useful in general cases, highlights local density variations
"""

function filter_highlight_gaussian(X::AbstractMatrix{<:Real}; kwargs...)
    dims = 2
    for (a, b) in kwargs
        if a == :dims
            dims = b
        end
    end
    size_of_matrix = size(X, 1)
    covariance_matrix = cov(X, dims = dims)
    inverse_covariance_matrix = inv(covariance_matrix)
    average = mean(X, dims = dims)
    centering = X.-average
    exp(-0.5 * centering'* inverse_covariance_matrix * centering) ./ sqrt(det(covariance_matrix) * (2π) ^ size_of_matrix)
end

function filter_general(X::AbstractMatrix{<:Real}; kwargs...)
    ϵ = 1.0
    for (a, b) in kwargs
        if a == :ϵ
            ϵ = b
        end
    end
    pairwise_kernel_matrix = MultivariateStats.pairwise((a, b) -> exp(-(1 / ϵ) * norm(a - b) ^ 2), X)
    sum_over_matrix = sum(pairwise_kernel_matrix, dims = 2)
    vec(sum_over_matrix./sum(sum_over_matrix))
end

# for covering, we are gonna go with the balanced approached

function balanced_cover(vector::AbstractVector{<:Real}; kwargs...)
    no_of_intervals = 5
    allowed_overlap = 0.5
    for (a, b) in kwargs
        if a == :no_of_intervals
            no_of_intervals = b
        elseif a == :allowed_overlap
            allowed_overlap = b
        end
    end
    min_vector, max_vector = extrema(vector)
    range_of_vector = max_vector - min_vector
    interval_range = 1.0 / (no_of_intervals - (no_of_intervals - 1) * allowed_overlap)
    steps = interval_range * (1 - allowed_overlap)

    ranges_of_intervals_with_overlap = [let range_min = (a - 1) * steps;
                (range_min, (range_min + interval_range)) .* range_of_vector .+ min_vector
            end
            for a in 1:no_of_intervals]
    indexes_for_cover_elements = map(a -> findall(b -> a[1] <= b <= a[2], vector), ranges_of_intervals_with_overlap)
    return indexes_for_cover_elements
end

# For cluster function, we will use K-means method, for which we will just use the Clustering.kmeans function from Clustering in Julia

cluster_function = Clustering.kmeans

# Cluster selection functions
"""
first we define a helping function for finding the maximum k value
"""

function max_K(N::Int; kwargs...)
    maximal_K = 10
    for (a, b) in kwargs
        if a == :kmax
            maximal_K = b
        end
    end
    return min(floor(Int, N/2), maximal_K)
end

"""
The first clustering selection function will be the Silhouette method
"""
function silhouette_method(cluster_func::Function, matrix::AbstractMatrix{<:Real}, kmax::Int=10; kwargs...)

    size_of_matrix = size(matrix, 1)
    maximal_K = min(kmax, floor(Int, size_of_matrix / 2))

    scores_and_assignments = []

    for k in 2:maximal_K
        clusters = try
            # Ensure clustering is applied to transposed matrix
            Clustering.kmeans(matrix', k; maxiter=300, display=:none)
        catch e
            println("ERROR in clustering with k=$k: $e")
            continue
        end
    
        # Ensure the assignments match the number of rows
        if length(clusters.assignments) != size(matrix, 1)
            println("ERROR: Cluster assignments length does not match submatrix size.")
            continue
        end

        # Normalize matrix along columns
        matrix_sample = matrix[:, :]
        normalized = (matrix_sample .- mean(matrix_sample, dims=1)) ./ std(matrix_sample, dims=1)

        # Compute pairwise distances
        pairwise_distances = try
            Distances.pairwise(Distances.Euclidean(), Float64.(normalized), dims=1)
        catch e
            println("ERROR in computing pairwise distances: $e")
            continue
        end

        # Compute silhouette scores
        silhouettes = try
            Clustering.silhouettes(clusters, pairwise_distances)
        catch e
            println("ERROR in computing silhouettes for k=$k: $e")
            continue
        end

        avg_silhouette = mean(silhouettes)
        push!(scores_and_assignments, (avg_silhouette, Clustering.assignments(clusters)))
    end

    # Select the clustering with the highest average silhouette score
    if isempty(scores_and_assignments)
        println("ERROR: No valid clusters found for silhouette method.")
        return []
    end

    max_score_index = argmax(map(first, scores_and_assignments))
    return scores_and_assignments[max_score_index][2]
end

"""
The second clustering selection function will be the Canopy method, which is computationally more expensive,
    however, it is better for large data sets and more precise.
    It also takes a bit of tuning of the borders, so it is a bit more complicated
"""

function canopy_method(cluster_func::Function, measurement_func::Function, matrix::AbstractMatrix{<:Real}, T_1::Real, T_2::Real; kwargs...)
    no_of_points = size(matrix, 2)
    remaining_points_indices = collect(1:no_of_points)
    index_of_cluster = 0
    labels_of_points = fill(0, no_of_points)
    distances_between_points = measurement_func(matrix; kwargs...)

    while !isempty(remaining_points_indices)
        index_of_cluster += 1
        center_point_random_index = remaining_points_indices[1]
        center_point_random = matrix[:, center_point_random_index]
        canopy_new = []

        for index in remaining_points_indices
            point_new = matrix[:, index]
            distance_between_new_point_and_center_point = distances_between_points[center_point_random_index, index]
            if distance_between_new_point_and_center_point <= T_1
                push!(canopy_new, index)
            end
        end
        remaining_points_indices = filter(index -> distances_between_points[center_point_random_index, index] > T_2, remaining_points_indices)

        if length(canopy) > 1
            canopy_points = view(matrix, :, canopy)
            perform_clustering = cluster_func(canopy_points; kwargs...)
            for (index_of_point, index_of_point_in_cluster) in zip(canopy, perform_clustering)
                if labels[index_of_point] == 0
                    labels[index_of_point] = index_of_cluster + index_of_point_in_cluster - 1
                end
            end
            index_of_cluster += maximum(perform_clustering) - 1
        else
            labels[canopy[1]] = index_of_cluster
        end
    end
    return labels_of_points
end

# layout of graph

constant_layout(mapper_data::Mapper) = (mapper_data.centers_of_patches[1,:], mapper_data.centers_of_patches[2,:])

function mapper(X::AbstractMatrix{<:Real}, filter, cover, cluster, cluster_selection; kwargs...)

    # first we calculate the filter range and make a cover of it
    filtering = filter(X; kwargs...)

    covering = cover(filtering; kwargs...)
    for interval in covering
        if any(index -> index > size(X, 1), interval)
            println("ERROR: Out-of-bounds interval: ", interval)
        end
    end
    # using these cover elements we create patches using clustering algorithm we want
    create_patches = Vector{Int}[]

    for elements in covering
        # if only one point, add immediately to the list
        if isempty(elements)
            continue
        end

        if length(elements) == 1
            push!(create_patches, elements)
            continue
        end

        # select the clusters and store the labels
        selected_clusters = cluster_selection(cluster, view(X, elements, :); kwargs...)
        unique_selected_clusters = unique(selected_clusters)
        for labels in unique_selected_clusters
            indices = elements[findall(isequal(labels), selected_clusters)]
            push!(create_patches, indices)
        end
    end

    # find overlapping patches using adjacency matrix and combine patches
    no_of_patches = length(create_patches)
    adjacency_matrix = SparseArrays.spzeros(UInt8, no_of_patches, no_of_patches)
    
    for a in 1:no_of_patches
        for b in (a+1):no_of_patches
            possible_overlap = intersect(create_patches[a], create_patches[b])

            # if there is overlap, add true to the adjacency matrix
            if length(possible_overlap) > 0
                adjacency_matrix[a, b] = 0x01
                adjacency_matrix[b, a] = 0x01
            end
        end
    end
    # we just need to find the centers of the calculated covering patches and return the corresponding structure
    list_of_centers = hcat((mean(view(X', :, patch), dims = 2) for patch in create_patches)...)
    return Mapper(adjacency_matrix, filtering, create_patches, list_of_centers)
end
using Plots
@recipe function f(mapper_data::Mapper; minimal_size = 10, maximal_size = 40)

    x_positions, y_positions = constant_layout(mapper_data)

    # set image limits
    x_min, x_max = extrema(x_positions)
    y_min, y_max = extrema(y_positions)
    xlims --> (x_min - 0.2 * (x_max - x_min), x_max + 0.2 * (x_max - x_min))
    ylims --> (y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))

    # show 1-skeleton
    for i in 1:size(mapper_data.adjacency_matrix, 1)
        indexes = findall(a -> a > 0, view(mapper_data.adjacency_matrix, i, :))
        push!(indexes, i)
        @series begin
            seriestype := :path
            linewidth --> 2
            linecolor --> :blue
            label --> ""
            x_positions[indexes], y_positions[indexes]
        end
    end

    # calculate vertex attribues
    no_of_patches = length(mapper_data.found_patches)
    x_coordinates = zeros(no_of_patches)
    y_coordinates = zeros(no_of_patches)
    z_column = zeros(no_of_patches)
    matrix_size = fill(1, no_of_patches)
    for (i, p) in enumerate(mapper_data.found_patches)
        z_column[i] = mean(mapper_data.filter_range[p])
        matrix_size[i] = length(p)
        x_coordinates[i] = x_positions[i]
        y_coordinates[i] = y_positions[i]
    end
    manno = map(string, matrix_size)
    s_min, s_max = extrema(matrix_size)
    s_range = s_max - s_min
    matrix_size = (maximal_size-minimal_size).*(matrix_size .- s_min)./s_range .+ minimal_size

    # show nodes
    @series begin
        seriestype := :scatter
        markersize := matrix_size
        label --> ""
        markercolor := RGBA(0.7, 0.8, 1.0, 0.6)
        series_annotations := manno
        x_coordinates, y_coordinates
    end
end

# chair_16

chair = get_abstract_matrix("data/non-articulated/chairs/chairsPly/b16.ply")
chair_mapper = mapper(chair, z_coordinate, balanced_cover, cluster_function, silhouette_method, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(chair_mapper.found_patches))
mapper_chair_plot = plot(chair_mapper)
savefig(mapper_chair_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_chair")

centers_chair = chair_mapper.centers_of_patches
using Distances
dist_matrix_chair = pairwise(Euclidean(), centers_chair)
using Ripserer

result_chair = ripserer(dist_matrix_chair)
fig_chair = plot(result_chair, title="Persistence Diagrams for Mapper Complex")
savefig(fig_chair, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram_chair")

# ant_16

ant = get_abstract_matrix("data/articulated/ants/antsPly/16.ply")
ant_mapper = mapper(ant, z_coordinate, balanced_cover, cluster_function, silhouette_method, no_of_intervals = 7, allowed_overlap = 0.7, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(ant_mapper.found_patches))
mapper_ant_plot = plot(ant_mapper)
savefig(mapper_ant_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_ant")

centers_ant = ant_mapper.centers_of_patches
using Distances
dist_matrix_ant = pairwise(Euclidean(), centers_ant)
using Ripserer

result_ant = ripserer(dist_matrix_ant)
fig_ant = plot(result_ant, title="Persistence Diagrams for Mapper Complex")
savefig(fig_ant, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram__ant")

# airplane_16

airplane = get_abstract_matrix("data/non-articulated/airplanes/airplanesPly/b16.ply")
airplane_mapper = mapper(airplane, z_coordinate, balanced_cover, cluster_function, silhouette_method, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(airplane_mapper.found_patches))
mapper_airplane_plot = plot(airplane_mapper)
savefig(mapper_airplane_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_airplane")

centers_airplane = airplane_mapper.centers_of_patches
using Distances
dist_matrix_airplane = pairwise(Euclidean(), centers_airplane)
using Ripserer

result_airplane = ripserer(dist_matrix_airplane)
fig_airplane = plot(result_airplane, title="Persistence Diagrams for Mapper Complex")
savefig(fig_airplane, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram_airplane")

# airplane_1

airplane1 = get_abstract_matrix("data/non-articulated/airplanes/airplanesPly/b1.ply")
airplane_mapper1 = mapper(airplane1, z_coordinate, balanced_cover, cluster_function, no_of_intervals=7, allowed_overlap=0.5, silhouette_method, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(airplane_mapper1.found_patches))
mapper_airplane1_plot = plot(airplane_mapper1)
savefig(mapper_airplane1_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_airplane1")

centers_airplane1 = airplane_mapper1.centers_of_patches
using Distances
dist_matrix_airplane1 = pairwise(Euclidean(), centers_airplane1)
using Ripserer

result_airplane1 = ripserer(dist_matrix_airplane1)
fig_airplane1 = plot(result_airplane1, title="Persistence Diagrams for Mapper Complex")
savefig(fig_airplane1, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram_airplane1")

# airplane_2

airplane2 = get_abstract_matrix("data/non-articulated/airplanes/airplanesPly/b2.ply")
airplane_mapper2 = mapper(airplane2, z_coordinate, balanced_cover, cluster_function, silhouette_method, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(airplane_mapper2.found_patches))
mapper_airplane2_plot = plot(airplane_mapper2)
savefig(mapper_airplane2_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_airplane2")

centers_airplane2 = airplane_mapper2.centers_of_patches
using Distances
dist_matrix_airplane2 = pairwise(Euclidean(), centers_airplane2)
using Ripserer

result_airplane2 = ripserer(dist_matrix_airplane2)
fig_airplane2 = plot(result_airplane2, title="Persistence Diagrams for Mapper Complex")
savefig(fig_airplane2, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram_airplane2")

# airplane_3

airplane3 = get_abstract_matrix("data/non-articulated/airplanes/airplanesPly/b3.ply")
airplane_mapper3 = mapper(airplane3, z_coordinate, balanced_cover, cluster_function, silhouette_method, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(airplane_mapper3.found_patches))
mapper_airplane3_plot = plot(airplane_mapper3)
savefig(mapper_airplane3_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_airplane3")

centers_airplane3 = airplane_mapper3.centers_of_patches
using Distances
dist_matrix_airplane3 = pairwise(Euclidean(), centers_airplane3)
using Ripserer

result_airplane3 = ripserer(dist_matrix_airplane3)
fig_airplane3 = plot(result_airplane3, title="Persistence Diagrams for Mapper Complex")
savefig(fig_airplane3, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram_airplane3")

# airplane_4

airplane4 = get_abstract_matrix("data/non-articulated/airplanes/airplanesPly/b4.ply")
airplane_mapper4 = mapper(airplane4, z_coordinate, balanced_cover, cluster_function, silhouette_method, ϵ = 1.0, measurement_func = point_distances2)
println("no_of_patches: ", length(airplane_mapper4.found_patches))
mapper_airplane4_plot = plot(airplane_mapper4)
savefig(mapper_airplane4_plot, "C:/Users/Nino/Desktop/tap_proj/project/mapper_airplane4")

centers_airplane4 = airplane_mapper4.centers_of_patches
using Distances
dist_matrix_airplane4 = pairwise(Euclidean(), centers_airplane4)
using Ripserer

result_airplane4 = ripserer(dist_matrix_airplane4)
fig_airplane4 = plot(result_airplane4, title="Persistence Diagrams for Mapper Complex")
savefig(fig_airplane4, "C:/Users/Nino/Desktop/tap_proj/project/persistence_diagram_airplane4")



