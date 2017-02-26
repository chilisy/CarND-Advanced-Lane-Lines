from function_lib import *
import global_setting
from moviepy.editor import VideoFileClip

# initialize TrackLine
global_setting.init()

folder = "../"
output_folder = "../output_images/"
#input_video_file = "challenge_video.mp4"
input_video_file = "project_video.mp4"
#input_video_file = "project_challenge.mp4"
output_video_file = output_folder + "processed_" + input_video_file

clip = VideoFileClip(folder + input_video_file)

# process the video
output_clip = clip.fl_image(process_image_for_video)

output_clip.write_videofile(output_video_file, audio=False)
