import torch.nn as nn
from sklearn.cluster import KMeans
from myUtils.config import Config
from myUtils.others import *


class DeepSpectral(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone_repo = self.config.method_config['repo']
        self.backbone_name = self.config.method_config['name']

        self.cluster_k = self.config.method_config["cluster_k"]
        self.eigen_k = self.config.method_config["eigen_vectors_k"]

        self.model = torch.hub.load(self.backbone_repo, self.backbone_name)
        self.binary_seg_threshold = self.config.binary_seg_threshold

        # self.cluster_n_cache = None

        self.crf_mark = self.config.crf_mark

        self.cache_pre_dir = f"cache/spectral/{self.backbone_name}_pre_seg/base/{self.cluster_k}-{self.eigen_k}"
        self.cache_crf_dir = f"cache/spectral/{self.backbone_name}_pre_seg/base-crf/{self.cluster_k}-{self.eigen_k}"
        self.cache_eigen_dir = f"cache/spectral/{self.backbone_name}_eigen_vt/base"

        if "vit" in self.backbone_name:
            self.model.fn = torch.nn.Identity()
            self.patch_size = self.model.patch_embed.patch_size
            self.num_heads = self.model.blocks[0].attn.num_heads

            self.hook_dict = {}

            def hook_func_qkv(module, input_x, output):
                self.hook_dict["qkv"] = output

            def hook_func_proj(module, input_x, output):
                self.hook_dict["proj"] = output

            self.model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_func_qkv)
            self.model._modules["blocks"][-1]._modules["attn"]._modules["proj"].register_forward_hook(hook_func_proj)

        else:
            raise ValueError(f"Unsupported name {self.backbone_name}")

        self.frozen_module_list = nn.ModuleList([self.model])
        self.trained_module_list = nn.ModuleList()

        self.g_sigma = self.config.method_config.get("g_sigma")

    def __forward(self, x):
        x = x.to(self.config.device)
        p = self.patch_size
        b, c, h, w = x.shape
        w_p_size, h_p_size = torch.div(w, p, rounding_mode="trunc").item(), torch.div(h, p,
                                                                                      rounding_mode="trunc").item()
        w_size, h_size = w_p_size * p, h_p_size * p

        p_size = (h_p_size, w_p_size)

        s = w_p_size * h_p_size

        x_resized = F.interpolate(x, size=[h_size, w_size], mode="bilinear")

        block_output = self.model.get_intermediate_layers(x_resized)[0]

        qkv = self.hook_dict["qkv"].reshape(b, s + 1, 3, -1).permute(2, 0, 1, 3)[:, :, 1:, :]
        proj = self.hook_dict["proj"][:, 1:, :]

        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, s, d]

        return proj, q, k, v, p_size

    def forward(self, x):
        return self.__forward(x)

    @staticmethod
    def __get_aff(feats_i):
        affinity_matrix = feats_i @ feats_i.T  # [h*w, h*w]

        affinity_matrix[affinity_matrix < 0] = 0

        affinity_matrix.fill_diagonal_(0)

        return affinity_matrix

    def __get_eigen(self, feats_i, x_i):
        c, h, w = x_i.shape
        h_p_size, w_p_size, d = feats_i.shape
        t = h_p_size * w_p_size

        feats_i = feats_i.reshape(t, d)

        feats_i = F.normalize(feats_i, dim=-1)

        affinity_matrix = self.__get_aff(feats_i)

        affinity_matrix_ = F.interpolate(affinity_matrix.unsqueeze(0).unsqueeze(0), size=(512, 512), mode="nearest")

        lap_ar_norm = get_normed_laplacian_matrix(affinity_matrix)
        eigen_values, eigen_vectors = torch.linalg.eigh(lap_ar_norm)

        return eigen_vectors, affinity_matrix_.squeeze(0)

    def __get_vector(self, eigen_vectors):
        eigen_num = self.config.method_config.get("eigen_vectors_k")
        cluster_k = self.config.method_config.get("cluster_k")

        if cluster_k == 2 and eigen_num == 1:
            # salient detection
            eigen_vector = eigen_vectors[:, 1:2]
        else:
            # spectral clustering
            eigen_vector = eigen_vectors[:, :eigen_num]

        return eigen_vector

    def __get_cluster(self, return_dict, eigen_vectors, batch_idx, feature_map):
        _, h_p_size, w_p_size, _ = feature_map.shape
        t = h_p_size * w_p_size

        eigen_vector = self.__get_vector(eigen_vectors)

        len_eigen = eigen_vector.shape[-1]

        eigen_vector = eigen_vector.reshape(1, h_p_size, w_p_size, -1).permute(3, 0, 1, 2)

        cluster_k = self.config.method_config.get("cluster_k")
        eigen_num = self.config.method_config.get("eigen_vectors_k")

        if cluster_k == 2 and eigen_num == 1:
            # salient detection
            eigen_vector = eigen_vector.permute(1, 2, 3, 0).reshape(-1, len_eigen)
            pre_seg = eigen_vector.detach().clone()
            pre_seg[pre_seg > 0] = 1
            pre_seg[pre_seg <= 0] = 0
        else:
            # spectral clustering
            eigen_vector_np = eigen_vector.reshape(-1, t).T.cpu().numpy()

            cluster_method = KMeans(n_clusters=cluster_k, n_init="auto",
                                    random_state=self.config.random_seed)
            cluster_method.fit(eigen_vector_np)
            pre_seg = cluster_method.predict(eigen_vector_np)

            pre_seg = torch.from_numpy(pre_seg)

        pre_seg = pre_seg.reshape(1, 1, h_p_size, w_p_size).to(self.config.device).long()

        return pre_seg

    def get_seg(self, return_dict):
        x = return_dict[0]['x'].to(self.config.device)

        b, c, h, w = x.shape

        proj, q, k, v, p_size = self.forward(x)

        h_p_size, w_p_size = p_size
        return_dict[0]["patched_size"] = [h_p_size, w_p_size]
        feature_map = k.reshape(b, h_p_size, w_p_size, -1)

        _, h_p_size, w_p_size, _ = feature_map.shape
        t = h_p_size * w_p_size

        return_dict[0]['pre_seg'] = []
        return_dict[0]["affinity_matrix"] = []
        return_dict[0]['eigen_vector'] = []

        for i in range(b):
            eigen_vectors, affinity_matrix_vis = self.__get_eigen(feature_map[i], x[i])
            return_dict[0]['eigen_vector'].append(eigen_vectors.reshape(1, h_p_size, w_p_size, -1)[..., :100])

            affinity_matrix_vis = (normalize_01_tensor(affinity_matrix_vis) * 255).byte()

            return_dict[0]["affinity_matrix"].append(affinity_matrix_vis)

            pre_seg = self.__get_cluster(return_dict, eigen_vectors, i, feature_map)

            pre_seg = resize_label_map(pre_seg, (h, w))

            return_dict[0]['pre_seg'].append(pre_seg.unsqueeze(0).unsqueeze(0))

        return_dict[0]["affinity_matrix"] = torch.cat(return_dict[0]["affinity_matrix"], dim=0)
        return_dict[0]['eigen_vector'] = torch.cat(return_dict[0]['eigen_vector'], dim=0)
        return_dict[0]['pre_seg'] = torch.cat(return_dict[0]['pre_seg'], dim=0).permute(0, 2, 3, 1)
        return_dict[0]['pre_seg_raw'] = torch.clone(return_dict[0]['pre_seg'].detach())

        return return_dict
