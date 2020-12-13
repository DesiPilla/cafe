import cv2
import glob
from utils import load_image, get_face, plot_multiple_with_box


friends = glob.glob('friends/fel.jpg')
print("{} images found.".format(len(friends)))
print(friends)

# Load friend images
friend_imgs = [load_image(f) for f in friends]

# Extract friend faces
faces = [get_face(img) for img in friend_imgs]
friend_faces = [f[0] for f in faces]
face_patches = [f[1] for f in faces]

# Plot images with face patches
plot_multiple_with_box(friend_imgs, face_patches)

# Save face to `face_only` directory for future use
face_only_paths = ['friends/face_only/' + f.split('/')[-1] for f in friends]

for i in range(len(friends)):
    img = friend_faces[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(face_only_paths[i], img)