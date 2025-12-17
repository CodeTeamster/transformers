# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..vit.configuration_vit import ViTConfig, ViTOnnxConfig
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEmbeddings,
    ViTEncoder,
    ViTForImageClassification,
    ViTForMaskedImageModeling,
    ViTIntermediate,
    ViTLayer,
    ViTModel,
    ViTOutput,
    ViTPatchEmbeddings,
    ViTPooler,
    ViTPreTrainedModel,
    ViTSelfAttention,
    ViTSelfOutput,
)


class ViTRDConfig(ViTConfig):
    pass


class ViTRDOnnxConfig(ViTOnnxConfig):
    pass


class ViTRDEmbeddings(ViTEmbeddings):
    pass


class ViTRDPatchEmbeddings(ViTPatchEmbeddings):
    pass


class ViTRDSelfAttention(ViTSelfAttention):
    pass


class ViTRDSelfOutput(ViTSelfOutput):
    pass


class ViTRDAttention(ViTAttention):
    pass


class ViTRDIntermediate(ViTIntermediate):
    pass


class ViTRDOutput(ViTOutput):
    pass


class ViTRDLayer(ViTLayer):
    pass


class ViTRDEncoder(ViTEncoder):
    pass


class ViTRDPreTrainedModel(ViTPreTrainedModel):
    pass


class ViTRDModel(ViTModel):
    pass


class ViTRDPooler(ViTPooler):
    pass


class ViTRDForMaskedImageModeling(ViTForMaskedImageModeling):
    pass


class ViTRDForImageClassification(ViTForImageClassification):
    pass


__all__ = [
    "ViTRDConfig",
    "ViTRDOnnxConfig",
    "ViTRDForImageClassification",
    "ViTRDForMaskedImageModeling",
    "ViTRDModel",
    "ViTRDPreTrainedModel",
]
