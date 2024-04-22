# ./mp4_to_frames.sh movie.mp4 savepath
mkdir -p $2
ffmpeg -r 1 -i $1 -r 1 $2/frame_%03d.jpg