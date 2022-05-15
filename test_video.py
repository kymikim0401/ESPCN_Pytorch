import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from ESPCN_model import ESPCN

if __name__ == "__main__":
    upscale = 3
    model = ESPCN(3).cuda()
    model.load_state_dict(torch.load('checkpoints/model_relu.pth'))

    test_vid_path = 'test_video/'
    videos_list = [video for video in os.listdir(test_vid_path)]
    result_path = 'test_video_result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    test_vid = videos_list[0]
    test_vid_capture = cv2.VideoCapture(test_vid_path + test_vid)
    fps = test_vid_capture.get(cv2.CAP_PROP_FPS)
    size = (int(test_vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * upscale),
            int(test_vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * upscale)
    output_name = result_path + test_vid.split('.')[0] + '.avi'
    test_vid_writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, size)

    success, frame = test_vid_capture.read()
    while success:
        frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('YCbCr')
        y, cb, cr = frame_img.split()
        frame_img_y = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        out = model.forward(frame_img_y.cuda())
        out = out.detach().cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.Resampling.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.Resampling.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        test_vid_writer.write(out_img)

        success, frame = test_vid_capture.read()




