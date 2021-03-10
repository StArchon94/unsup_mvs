import numpy as np
import open3d as o3d


def load_cam_ext(file):
    """ read camera txt file """
    cam = np.empty((3, 4))
    data = file.read().split()
    for i in range(3):
        for j in range(4):
            cam[i, j] = data[4 * i + j + 1]
    return cam


def compute_center_from_transformation(R, t):
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def cameras_lineset(Rs, ts, size=10, color=(0.5, 0.5, 0), connect_cams=False):
    points = []
    lines = []
    colors = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        C0 = compute_center_from_transformation(R, t).reshape(3, 1)
        cam_points = []
        cam_points.append(C0)
        cam_points.append(C0 + R.T @ np.array([-size, -size, 3.0 * size]).reshape(3, 1))
        cam_points.append(C0 + R.T @ np.array([+size, -size, 3.0 * size]).reshape(3, 1))
        cam_points.append(C0 + R.T @ np.array([+size, +size, 3.0 * size]).reshape(3, 1))
        cam_points.append(C0 + R.T @ np.array([-size, +size, 3.0 * size]).reshape(3, 1))
        cam_points = np.concatenate([pt.reshape(1, 3) for pt in cam_points], axis=0)
        cam_lines = np.array([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)])

        points.extend(cam_points)
        if connect_cams and len(lines):
            cam_lines = np.vstack((cam_lines, [-5, 0]))
        lines.extend(i * 5 + cam_lines)
        colors.extend([color for _ in range(len(cam_lines))])

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return ls


if __name__ == '__main__':
    scan = 'scan10'
    ref_no = 24
    default = set(np.load(f'/home/slin/Documents/outputs/mvs/exp1/tests/50000/{scan}/depths_mvsnet/{ref_no:08d}_view_comb.npy'))
    # comp = set(np.load(f'/home/slin/Documents/outputs/mvs/exp1/tests/50000_nn/{scan}/depths_mvsnet/{ref_no:08d}_view_comb.npy'))
    comp = set(np.load(f'/home/slin/Documents/outputs/mvs/exp1/tests/50000_all/{scan}/depths_mvsnet/{ref_no:08d}_view_comb.npy'))
    other = set(range(49)) - {ref_no} - default - comp
    common = default & comp
    default -= common
    comp -= common
    cam_dir = f'/home/slin/Documents/datasets/dtu/test/{scan}/cams/'
    cam_size = 16

    ref_cam_pose = load_cam_ext(open(cam_dir + f'{ref_no:08d}_cam.txt'))
    ref_cam_ls = cameras_lineset(ref_cam_pose[None, :, :3], ref_cam_pose[None, :, 3], cam_size, (1, 0, 0))
    ls_list = [ref_cam_ls]

    if common:
        common_cam_poses = []
        for common_no in common:
            common_cam_poses.append(load_cam_ext(open(cam_dir + f'{common_no:08d}_cam.txt')))
        common_cam_poses = np.stack(common_cam_poses)
        common_cam_ls = cameras_lineset(common_cam_poses[..., :3], common_cam_poses[..., 3], cam_size, (1, 0.5, 0))
        ls_list.append(common_cam_ls)

    if default:
        default_cam_poses = []
        for default_no in default:
            default_cam_poses.append(load_cam_ext(open(cam_dir + f'{default_no:08d}_cam.txt')))
        default_cam_poses = np.stack(default_cam_poses)
        default_cam_ls = cameras_lineset(default_cam_poses[..., :3], default_cam_poses[..., 3], cam_size, (0, 0, 1))
        ls_list.append(default_cam_ls)

    if comp:
        comp_cam_poses = []
        for comp_no in comp:
            comp_cam_poses.append(load_cam_ext(open(cam_dir + f'{comp_no:08d}_cam.txt')))
        comp_cam_poses = np.stack(comp_cam_poses)
        comp_cam_ls = cameras_lineset(comp_cam_poses[..., :3], comp_cam_poses[..., 3], cam_size, (0, 1, 0))
        ls_list.append(comp_cam_ls)

    other_cam_poses = []
    for other_no in other:
        other_cam_poses.append(load_cam_ext(open(cam_dir + f'{other_no:08d}_cam.txt')))
    other_cam_poses = np.stack(other_cam_poses)
    other_cam_ls = cameras_lineset(other_cam_poses[..., :3], other_cam_poses[..., 3], cam_size, (0, 0, 0))
    ls_list.append(other_cam_ls)

    o3d.visualization.draw_geometries(ls_list)
