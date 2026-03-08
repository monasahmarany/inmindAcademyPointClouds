import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):

    # Downsample to reduce noise and computation
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals using a search radius of 2x the voxel size
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Compute FPFH features using a search radius of 5x the voxel size
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd_down, fpfh


def global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh,
    target_fpfh,
    voxel_size: float,
) -> np.ndarray:
    distance_threshold = voxel_size * 4.0
    print(f"[Global] RANSAC with distance threshold = {distance_threshold:.4f}")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=10,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.75),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )

    print(f"[Global] Fitness={result.fitness:.4f}, Inlier RMSE={result.inlier_rmse:.4f}")
    return result.transformation


def local_refinement(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    distance_threshold = voxel_size * 0.4
    print(f"[Local]  ICP point-to-plane with distance threshold = {distance_threshold:.4f}")

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100,
        ),
    )

    print(f"[Local]  Fitness={result.fitness:.4f}, Inlier RMSE={result.inlier_rmse:.4f}")
    return result.transformation


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:

    voxel_size = 0.08 # metres; tune this for your scene scale

    print("=== Stage 1: Preprocessing ===")
    source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)

    print("=== Stage 2: Global Registration (RANSAC + FPFH) ===")
    global_transform = global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )

    print("=== Stage 3: Local Refinement (Point-to-Plane ICP) ===")
    # ICP needs normals on the full-resolution clouds
    radius_normal = voxel_size * 2
    pcd1.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd2.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    refined_transform = local_refinement(pcd1, pcd2, global_transform, voxel_size)

    print("=== Registration complete ===")
    print("Final transformation matrix:")
    print(np.array2string(refined_transform, precision=4, suppress_small=True))

    return refined_transform