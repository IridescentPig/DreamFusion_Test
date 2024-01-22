import torch
import igl

def warp_observation_to_canonical(points, verts, faces, joint_trans_mat, verts_weights, return_signed_dis=False):
    """
    Warp the points of observation space to canonical space
    Args:
        points: (N, 3)
        verts: (V, 3)
        faces: (F, 3)
        joint_trans_mat: (16, 4, 4)
        verts_weights: (V, 16)
    Return:
        trans_mat: (N, 4, 4)
    """
    signed_dis, idxs, closests = igl.signed_distance(points.detach().cpu().numpy(), verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
    # idxs: (N, )
    # closests: (N, 3)
    closests = torch.from_numpy(closests).float()
    closest_tri = verts[faces[idxs]] # closest_tri: (N, 3, 3)
    v0v1 = closest_tri[:, 1] - closest_tri[:, 0] # v0v1: (N, 3)
    v0v2 = closest_tri[:, 2] - closest_tri[:, 0] # v0v2: (N, 3)
    v1v2 = closest_tri[:, 2] - closest_tri[:, 1] # v1v2: (N, 3)
    v2v0 = closest_tri[:, 0] - closest_tri[:, 2] # v2v0: (N, 3)
    v1p = closests - closest_tri[:, 1] # v1p: (N, 3)
    v2p = closests - closest_tri[:, 2] # v2p: (N, 3)
    N = torch.cross(v0v1, v0v2) # N: (N, 3)
    denom = torch.bmm(N.unsqueeze(1), N.unsqueeze(2)).squeeze() # denom: (N, )
    C1 = torch.cross(v1v2, v1p) # C1: (N, 3)
    u = torch.bmm(N.unsqueeze(1), C1.unsqueeze(2)).squeeze() / denom # u: (N, )
    C2 = torch.cross(v2v0, v2p) # C2: (N, 3)
    v = torch.bmm(N.unsqueeze(1), C2.unsqueeze(2)).squeeze() / denom # v: (N, )
    w = 1 - u - v # w: (N, )
    barycentric = torch.stack([u, v, w], dim=1) # barycentric: (N, 3)
    points_weights = verts_weights[faces[idxs]] # points_weights: (N, 3, 16)
    points_weights = torch.bmm(barycentric.unsqueeze(1), points_weights).squeeze() # points_weights: (N, 16)

    trans_mat = torch.matmul(points_weights, joint_trans_mat.reshape(16, 16)) # trans_mat: (N, 16)
    signed_dis = torch.from_numpy(signed_dis).float()
    signed_dis = signed_dis - 0.005
    signed_dis = torch.relu(signed_dis)
    # coe = torch.exp(-signed_dis*10).unsqueeze(1) # coe: (N, 1)

    # identity_trans = torch.eye(4).repeat(points.shape[0], 1, 1) * 0.1
    # return torch.inverse(trans_mat.reshape(-1, 4, 4))
    # identity_trans = identity_trans.reshape(-1, 16)

    # trans_mat = coe * trans_mat + (1 - coe) * identity_trans
    
    if return_signed_dis:
        return trans_mat.reshape(-1, 4, 4), signed_dis
    else:
        return trans_mat.reshape(-1, 4, 4)