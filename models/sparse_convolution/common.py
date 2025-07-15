import collections

import torch.nn as nn
import MinkowskiEngine as ME

from enum import Enum
from typing import Union


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type: NormType, n_channels: int, D: int, bn_momentum: float = 0.1):
    """Convert NormType enum into minkowski norm

    Args:
        norm_type (NormType): NormType enum
        n_channels (int): number of channels
        D (int): dimension of input_
        bn_momentum (float, optional): norm momentum. Defaults to 0.1.

    Raises:
        ValueError: if NormType is not supported

    Returns:
        norm: Minkowski Engine norm corresponding to the norm type
    """
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels),
            ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum),
        )
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


def get_nonlinearity(non_type: str):
    if non_type == "ReLU":
        return ME.MinkowskiReLU()
    elif non_type == "ELU":
        return ME.MinkowskiELU()
    else:
        raise ValueError(f"Type {non_type}, not defined")


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS "

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Covert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.CUSTOM,
}

int_to_region_type = {
    0: ME.RegionType.HYPER_CUBE,
    1: ME.RegionType.HYPER_CROSS,
    2: ME.RegionType.CUSTOM,
}


def convert_region_type(region_type: int):
    """
    Convert the integer region_type to the corresponding RegionType enum object.
    """
    return int_to_region_type[region_type]


def convert_conv_type(
    conv_type: ConvType, kernel_size: Union[collections.Sequence, int], D: int
):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
        if D < 4:
            region_type = ME.RegionType.HYPER_CUBE
        else:
            axis_types = [
                ME.RegionType.HYPER_CUBE,
            ] * 3
            if D == 4:
                axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[collections.Sequence, int],
    stride: int = 1,
    dilation: int = 1,
    bias: bool = False,
    conv_type: ConvType = ConvType.HYPERCUBE,
    D: int = -1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def conv_tr(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[collections.Sequence, int],
    upsample_stride: int = 1,
    dilation: int = 1,
    bias: bool = False,
    conv_type: ConvType = ConvType.HYPERCUBE,
    D: int = -1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def avg_pool(
    kernel_size: Union[collections.Sequence, int],
    stride: int = 1,
    dilation: int = 1,
    conv_type: ConvType = ConvType.HYPERCUBE,
    in_coords_key=None,
    D: int = -1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiAvgPooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def avg_unpool(
    kernel_size: Union[collections.Sequence, int],
    stride: int = 1,
    dilation: int = 1,
    conv_type: ConvType = ConvType.HYPERCUBE,
    D: int = -1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiAvgUnpooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def sum_pool(
    kernel_size: Union[collections.Sequence, int],
    stride: int = 1,
    dilation: int = 1,
    conv_type: ConvType = ConvType.HYPERCUBE,
    D: int = -1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiSumPooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )
