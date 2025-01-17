import torch
import torch.nn.functional as F
import math


def get_svls_filter_3d(kernel_size=3, sigma=1, channels=4):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
    z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 3-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance + 1e-16))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    neighbors_sum = 1 - gaussian_kernel[1,1,1]
    gaussian_kernel[1,1,1] = neighbors_sum
    svls_kernel_3d = gaussian_kernel / neighbors_sum

    # Reshape to 3d depthwise convolutional weight
    svls_kernel_3d = svls_kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1, 1)
    svls_filter_3d = torch.nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=0)
    svls_filter_3d.weight.data = svls_kernel_3d
    svls_filter_3d.weight.requires_grad = False 
    return svls_filter_3d, svls_kernel_3d[0]

def get_svls_filter_3d_modified(kernel_size=3, sigma=1, channels=4, ratio=1.0):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
    z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 3-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance + 1e-16))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    neighbors_sum = 1 - gaussian_kernel[1,1,1]
    gaussian_kernel[1,1,1] = neighbors_sum * ratio
    svls_kernel_3d = gaussian_kernel / neighbors_sum

    # Reshape to 3d depthwise convolutional weight
    svls_kernel_3d = svls_kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1, 1)
    svls_filter_3d = torch.nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=0)
    svls_filter_3d.weight.data = svls_kernel_3d
    svls_filter_3d.weight.requires_grad = False 
    return svls_filter_3d, svls_kernel_3d[0]

class CELossWithSVLS(torch.nn.Module):
    def __init__(self, classes=None, sigma=1):
        super(CELossWithSVLS, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()

    def forward(self, inputs, labels):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()


class CELossWithLS(torch.nn.Module):
    def __init__(self, classes=None, smoothing=0.0):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()

    def forward(self, inputs, labels):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            smoothen_label = oh_labels * self.complement + self.smoothing / self.cls
        return (- smoothen_label * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

class CELossWithSVLS_V2(torch.nn.Module):
    def __init__(self, classes=None, sigma=1):
        super(CELossWithSVLS_V2, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()

    def forward(self, inputs, labels, skeleton):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()
            # constrain the label by the skeleton
            sum_skeleton = torch.sum(skeleton, dim=1, keepdim=True)
            constrained_svls_label = torch.where(
                sum_skeleton==1,
                oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float(),
                svls_labels)

        return (- constrained_svls_label * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

class CELossWithSVLS_V3(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, scale_factor=1.0):
        super(CELossWithSVLS_V3, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()
        self.scale_factor = scale_factor

    def forward(self, inputs, labels):
        with torch.no_grad():
            n, z, x, y = labels.shape

            aff = torch.FloatTensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]).expand(n, 3, 4)
            grid_up = F.affine_grid(aff, size=(n, 1, int(self.scale_factor * z),
                                                     int(self.scale_factor * x),
                                                     int(self.scale_factor * y))).cuda()
            grid_down = F.affine_grid(aff, size=(n, 1, z, x, y)).cuda()

            # up sample
            labels_up_sample = F.grid_sample(labels[:, None, ...].float(), grid_up, mode='nearest').squeeze(dim=1)

            # get svls label
            oh_labels = (labels_up_sample[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()

            # down sample
            svls_labels = F.grid_sample(svls_labels.float(), grid_down, mode='bilinear')

        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

class CELossWithSVLS_V4(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, alpha=0.1):
        super(CELossWithSVLS_V4, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()
        self.alpha = alpha

    def forward(self, inputs, labels):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()

            # find where to apply threshold
            svls_labels_target = (svls_labels * oh_labels).sum(dim=1, keepdim=True)

            # get the svls labels after threshold
            svls_labels_target_threshold = oh_labels * (1.0 - self.alpha)  # target part
            svls_labels_other_threshold = svls_labels * (oh_labels == 0) * self.alpha \
                                        / (1.0 - svls_labels_target + 1e-6)  # other part
                                        
            svls_labels_threshold = svls_labels_target_threshold + svls_labels_other_threshold

            # apply threshold
            svls_labels_threshold = torch.where(svls_labels_target >= (1.0 - self.alpha), svls_labels, svls_labels_threshold)
        return (- svls_labels_threshold * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

class CELossWithSVLS_V5(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, ratio=1.0):
        super(CELossWithSVLS_V5, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d_modified(sigma=sigma, channels=classes, ratio=ratio)
        self.svls_kernel = self.svls_kernel.cuda()

    def forward(self, inputs, labels):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

class CELossWithSVLS_V6(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, scale_factor=1.0):
        super(CELossWithSVLS_V6, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()
        self.scale_factor = scale_factor

    def forward(self, inputs, labels):
        # calculate crop padding
        d_, h_, w_ = labels.shape[-3:]
        padding_d, padding_h, padding_w = int((d_ - (d_ / self.scale_factor)) / 2), \
                                        int((h_ - (h_ / self.scale_factor)) / 2), \
                                        int((w_ - (w_ / self.scale_factor)) / 2)

        with torch.no_grad():
            # center crop labels
            labels = labels[..., padding_d:-padding_d, padding_h:-padding_h, padding_w:-padding_w]
            # interpolate to recover the spatial size
            labels = F.interpolate(labels[:, None, ...].float(), size=(d_, h_, w_), mode='nearest')[:, 0, ...]
            # get svls label for scaled label maps
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()

        # center crop model outputs
        inputs = inputs[..., padding_d:-padding_d, padding_h:-padding_h, padding_w:-padding_w]
        # interpolate to recover the spatial size
        inputs = F.interpolate(inputs, size=(d_, h_, w_), mode='trilinear')

        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

def get_gaussian_filter(kernel_size=3, sigma=1):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
    z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 3-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance + 1e-16))

    return gaussian_kernel

class CELossWithSVLS_VE(torch.nn.Module):
    def __init__(self, classes=None, sigma_dist=1, sigma_diff=1):
        super(CELossWithSVLS_VE, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.dist_weight = get_gaussian_filter(sigma=sigma_dist)
        self.dist_weight = self.dist_weight[None, None, None, None, None, ...].cuda()
        self.sigma_diff = sigma_diff

    def forward(self, inputs, labels, images):
        with torch.no_grad():
            # grayscale diff weights
            image_diff = F.pad(images, (1,1,1,1,1,1), mode='replicate') \
                            .unfold(2, size=3, step=1) \
                            .unfold(3, size=3, step=1) \
                            .unfold(4, size=3, step=1) - images[..., None, None, None]
            image_diff, _ = torch.max(image_diff, dim=1, keepdim=True)  # extract the max difference
            weights = (1./(2.*math.pi*self.sigma_diff**2 + 1e-16)) * torch.exp(
                                    -image_diff**2 / (2*self.sigma_diff**2 + 1e-16))
            del image_diff
            
            # elementwise poduct to combine diff_weights and dist_weight
            weights = weights * self.dist_weight

            # Make sure sum of values in gaussian kernel equals 1 and sum_neighbors == center
            weights = weights / torch.sum(weights, dim=(-3, -2, -1), keepdim=True)
            neighbors_sum = 1 - weights[..., 1:2,1:2,1:2] + 1e-16
            weights[..., 1:2,1:2,1:2] = neighbors_sum
            weights = weights / neighbors_sum
            del neighbors_sum

            # get label patches
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            oh_labels = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            label_oh_patches = F.pad(oh_labels, (1,1,1,1,1,1), mode='replicate') \
                                .unfold(2, size=3, step=1) \
                                .unfold(3, size=3, step=1) \
                                .unfold(4, size=3, step=1)

            # get smoothed labels
            label_svls_ve_ = label_oh_patches * weights / torch.sum(weights, dim=(-3, -2, -1), keepdim=True)
            label_svls_ve = torch.sum(label_svls_ve_, dim=(-3, -2, -1))
            del label_oh_patches, label_svls_ve_, weights
        return (- label_svls_ve * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

class CELossWithSVLS_V7(torch.nn.Module):
    def __init__(self, classes=None, sigma_dist=1, sigma_diff=1, scale_factor=1.0):
        super(CELossWithSVLS_V7, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.dist_weight = get_gaussian_filter(sigma=sigma_dist).half().cuda()
        self.dist_weight = self.dist_weight[None, None, None, None, None, ...].cuda()
        self.sigma_diff = sigma_diff
        self.scale_factor = scale_factor
        self.aff = torch.FloatTensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])

    def forward(self, inputs, labels, images):
        with torch.no_grad():
            images = images.half()
            labels = labels.half()

            n, z, x, y = labels.shape

            aff = self.aff.expand(n, 3, 4)

            # up sample
            grid_up = F.affine_grid(aff, size=(n, 1, int(self.scale_factor * z),
                                                     int(self.scale_factor * x),
                                                     int(self.scale_factor * y))).half().cuda()
            labels = F.grid_sample(labels[:, None, ...], grid_up, mode='nearest').squeeze(dim=1)
            image_diff = F.grid_sample(images[:, 1:2, ...], grid_up, mode='bilinear')  # only use the first channel
            del grid_up
            torch.cuda.empty_cache()

            # grayscale diff weights
            image_diff = F.pad(image_diff, (1,1,1,1,1,1), mode='replicate') \
                            .unfold(2, size=3, step=1) \
                            .unfold(3, size=3, step=1) \
                            .unfold(4, size=3, step=1) - image_diff[..., None, None, None]
            # image_diff, _ = torch.max(image_diff, dim=1, keepdim=True)  # extract the max difference
            # del _
            # torch.cuda.empty_cache()

            # image_diff = image_diff.float()
            # print(torch.sum(torch.isnan(image_diff)))

            weights = (1./(2.*math.pi*self.sigma_diff**2 + 1e-16)) * torch.exp(
                                    -image_diff**2 / (2*self.sigma_diff**2 + 1e-16))
            # print(torch.sum(torch.isnan(weights)))
            del image_diff
            torch.cuda.empty_cache()

            # print("dist加权之前卷积核之和为0的个数：", torch.sum(torch.sum(weights, dim=(-3, -2, -1), keepdim=True)==0))
            
            # elementwise poduct to combine diff_weights and dist_weight
            weights = weights * self.dist_weight
            # print(torch.sum(torch.isnan(weights)))

            # Make sure sum of values in gaussian kernel equals 1 and sum_neighbors == center
            # print(torch.sum(torch.sum(weights, dim=(-3, -2, -1), keepdim=True)==0))
            weights = weights / torch.sum(weights, dim=(-3, -2, -1), keepdim=True)
            # print('after first divide:', torch.sum(torch.isnan(weights)))
            neighbors_sum = 1 - weights[..., 1:2,1:2,1:2] + 1e-6
            # print(torch.sum(neighbors_sum==0))
            weights[..., 1:2,1:2,1:2] = neighbors_sum
            weights = weights / neighbors_sum
            # print(torch.sum(torch.isnan(weights)))
            del neighbors_sum
            torch.cuda.empty_cache()
            weights = weights / torch.sum(weights, dim=(-3, -2, -1), keepdim=True)
            # print(torch.sum(torch.isnan(weights)))

            # get label patches
            labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = labels.shape
            labels = labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).half()
            labels = F.pad(labels, (1,1,1,1,1,1), mode='replicate') \
                                .unfold(2, size=3, step=1) \
                                .unfold(3, size=3, step=1) \
                                .unfold(4, size=3, step=1)

            # get smoothed labels
            labels = torch.sum(labels * weights, dim=(-3, -2, -1))
            # print(torch.sum(torch.isnan(labels)))
            del weights
            torch.cuda.empty_cache()

            # down sample
            grid_down = F.affine_grid(aff, size=(n, 1, z, x, y)).half().cuda()
            labels = F.grid_sample(labels, grid_down, mode='bilinear')
            # print(torch.sum(torch.isnan(labels)))
            del grid_down
            torch.cuda.empty_cache()
            labels = labels.float()
        return (- labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()
