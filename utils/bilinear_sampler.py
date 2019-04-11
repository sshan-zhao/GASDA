# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Modifications Clement Godard.
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
# ==============================================================================

import torch

def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):

        rep = x.view(-1,1).repeat(1, n_repeats)
        return rep.reshape(-1)

    def _interpolate(im, x, y):

        # handle both texture border types
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size = 1
            pad = torch.nn.ConstantPad2d(1, 0)
            im = pad(im)
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = torch.clamp(x, 0.0, _width_f -1 + 2 * _edge_size)

        x0_f = torch.floor(x)
        y0_f = torch.floor(y)
        x1_f = x0_f + 1

        x0 = x0_f.int()
        y0 = y0_f.int()
        x1 = torch.clamp(x1_f, max = _width_f -1 + 2 * _edge_size).int()

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(torch.arange(_num_batch) * dim1, _height * _width).int().cuda()
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_per = im.permute(0, 2, 3, 1)
        im_flat = torch.reshape(im_per, (-1, _num_channels))

        pix_l = im_flat[idx_l.long(),:]
        pix_r = im_flat[idx_r.long(),:]

        weight_l = (x1_f - x).view(-1, 1)
        weight_r = (x - x0_f).view(-1, 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):

        x_t_flat = torch.arange(0, _width).view(1, -1).repeat(_height, 1).float().cuda()
        x_t_flat = x_t_flat.view(1, _height, _width).repeat(_num_batch, 1, 1).view(-1)
        y_t_flat = torch.arange(0, _height).view(-1, 1).repeat(1, _width).float().cuda()
        y_t_flat = y_t_flat.view(1, _height, _width).repeat(_num_batch, 1, 1).view(-1)

        x_t_flat += x_offset.view(-1) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = torch.reshape(input_transformed, [_num_batch, _height, _width, _num_channels]).permute(0, 3, 1, 2)

        return output

    _num_batch    = input_images.shape[0]
    _height       = input_images.shape[2]
    _width        = input_images.shape[3]
    _num_channels = input_images.shape[1]

    _height_f = float(_height)
    _width_f  = float(_width)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)
    return output
