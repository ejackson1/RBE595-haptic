# 3D AABB Tree Generation & Closest Point Estimation
This project was designed for RBE595-Haptic & Robot Interaction. Only one file is necessary to run the code- rbe533.py. 
The purpose of this project was to generate a 3D Axis Aligned Bounding Box (AABB) tree around an arbitrary object. The Standford bunny was used as a proof of concept. Further, a Depth First Search (DPS) determines the closest centroid of the AABB leaflet to the origin point (not shown in OpenGL simulation). This distance value is output to the terminal. 
The AABB tree and box centroids are only calcualted once initially. After, their centroids are multiplied by a transformation matrix to determine their position. The arrow keys on the keyboard can be used to translate the Standford bunny. Additionally, the bunny can be programmed to stop spinning by commenting out some lines.

## Dependencies
The following dependencies can all be downladed by the following set of commands.
`python -m pip install pygame`
`python -m pip install OpenGL`
`python -m pip install pywavefront`
`python -m pip install operator`
`python -m pip install numpy`
`python -m pip install trelib`

## Screenshots and Current Limitations
The depth/level of the tree can be controlled in the code. Currently, this method only supports a tree depth of 7. The resolution, an arbitrary value used to discretly "slide" across a particular axis of the respective box in the AABB tree, must be lowered to increase the depth. Otherwise, the length dimension of the smallest (and deepest) box will be greater than the resolution. A further implementation would be to create a dynamic resolution that changes with box length. 
