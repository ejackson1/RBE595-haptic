### RBE533 Haptic Interaction ###
# This code was generated for RBE533's Final Project
# Created By: Edward Jackson
# 10/10/2022


import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront
from operator import mul
import numpy as np
from treelib import Node, Tree

scene = pywavefront.Wavefront('bunny.obj', collect_faces=True) # import bun bun 

scene_box = (scene.vertices[0], scene.vertices[0])
for vertex in scene.vertices:
    min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
    max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
    scene_box = (min_v, max_v)

scene_size     = [scene_box[1][i]-scene_box[0][i] for i in range(3)]
max_scene_size = max(scene_size)
scaled_size    = 5
scene_scale    = [scaled_size/max_scene_size for i in range(3)]
scene_trans    = [-(scene_box[1][i]+scene_box[0][i])/2 for i in range(3)]

# Debug Info
print(f"scene_box: {scene_box}")
print(f"scene_scale: {scene_scale}")
print(f"scene_trans: {scene_trans}")

print(f"type:{type(scene_box[0][0])}")

def Model():
    glPushMatrix()
    glScalef(*scene_scale)
    glTranslatef(*scene_trans)

    for mesh in scene.mesh_list:    
        glBegin(GL_TRIANGLES)
        for face in mesh.faces: ## connections to form triangle. Indecies of verticies
            for vertex_i in face:
                glVertex3f(*scene.vertices[vertex_i])
        glEnd()

    glPopMatrix()



verticies = (
    tuple(map(mul, (1, 1, -1) , (abs(scene_box[1][0]), abs(scene_box[0][1]), abs(scene_box[0][2])))),
    tuple(map(mul, (1, 1, -1) , (abs(scene_box[1][0]), abs(scene_box[1][1]), abs(scene_box[0][2])))),
    tuple(map(mul, (-1, 1, -1) , (abs(scene_box[0][0]), abs(scene_box[1][1]), abs(scene_box[0][2])))),
    tuple(map(mul, (-1, 1, -1) , (abs(scene_box[0][0]), abs(scene_box[0][1]), abs(scene_box[0][2])))),
    
    tuple(map(mul, (1, 1, 1) , (abs(scene_box[1][0]), abs(scene_box[0][1]), abs(scene_box[1][2])))),
    tuple(map(mul, (1, 1, 1) , (abs(scene_box[1][0]), abs(scene_box[1][1]), abs(scene_box[1][2])))),
    tuple(map(mul, (-1, 1, 1) , (abs(scene_box[0][0]), abs(scene_box[0][1]), abs(scene_box[1][2])))),
    tuple(map(mul, (-1, 1, 1) , (abs(scene_box[0][0]), abs(scene_box[1][1]), abs(scene_box[1][2]))))
    )

print(verticies)
edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

def Cube():
    glPushMatrix()
    glScalef(*scene_scale)
    glTranslatef(*scene_trans)

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])

    glEnd()
    glPopMatrix()

def vertsCount(verts, k, axis):
    """
    verts: [xyz],[xyz].... inside of B
    k: moving slider along axis
    axis: 0,1,2 -> x,y,z
    Returns number of triangles in box up to k along axis
    """
    Nr = 0
    for vertex in verts:
        if vertex[axis] > k:
            Nr += 1

    return Nr


def vertsList(verts, k, axis):
    """
    Exact same function as vertsCount but creates a list of vertex coordinates for left and right children nodes
    We don't integrate it into vertsCount because it would become innefficient 
    """
    verts_r = []
    verts_l = []
    for vertex in verts:
        if vertex[axis] > k:
            verts_r.append(vertex)
        else:
            verts_l.append(vertex)

    return (verts_l, verts_r)


def SA_Helper(Bi):
    # Bi: [x,y,z],[x,y,z] of bounding box
    #print(f"Bi {Bi}")
    diff = np.subtract(Bi[0], Bi[1])
    #print(f"Diff: {diff}")
    x = diff[0]
    y = diff[1]
    z = diff[2]
    SA = 2*x*y + 2*x*z + 2*y*z
    return SA



    
def SAH(Bi, verts, k, axis):
    fcost_min = 10000000000000000000000 # init
    k_array = [0] * 3

    while k <= Bi[1][axis]: # while it hasnt finished reaching the end
        # Evaluate cost function
        SA = SA_Helper(Bi)
        Cr = vertsCount(verts, k, axis) # number of vertices to the right of k
        Cl = len(verts) - Cr # subtract the difference to find the left 
        
        # Calculate Surface Area of proposed left and right sided child nodes
        k_array[axis] = k
        Bi_l = (Bi[0], np.subtract(Bi[1], k_array).tolist()) 
        Bi_r = (np.add(Bi[0], k_array).tolist(), Bi[1])

        #print(f"Bi_l {Bi_l}, Bi_r {Bi_r}")
        
        SA_l = SA_Helper(Bi_l)
        SA_r = SA_Helper(Bi_r)

        # Cost Calculation & Comparison to previous minimum cost
        fcost = Cl * SA_l/SA + Cr * SA_r/SA
        if fcost_min > fcost:
            fcost_min = fcost
            k_best = k

            # Find best Bi_l and Bi_r with given k_best. Very low cost operation 
            k_array[axis] = k_best
            Bi_l = [Bi[0], np.subtract(Bi[1], k_array).tolist()]
            Bi_r = [np.add(Bi[0], k_array).tolist(), Bi[1]]

        k += 0.001

    return (fcost_min, k_best, Bi_l, Bi_r, Cl, Cr)


class AABB(object):
    def __init__(self, Bi, verts):
        self.Bi = Bi
        self.verts = verts



init = None
def BVHTree():

    ftree = Tree()
    ftree.create_node(0, 0, data=AABB(scene_box, scene.vertices))
    #ftree.create_node(1, 1, parent=0, data=AABB(scene_box, scene.vertices))
    #ftree.create_node(2, 2, parent=0, data=AABB(scene_box, scene.vertices))
    ftree.show()

    #cNode = ftree.get_node(2).data # how to extract tag
    print(ftree.leaves())
    
    while int(ftree.depth()) < 1: # only go 5 layers
        bottomLayer = ftree.leaves()
        lastPTag = int(bottomLayer[-1].tag)
        firstPTag = int(bottomLayer[0].tag)
        
        nodesCount = len(bottomLayer)
        # if lastPTag == firstPTag: ## root
        #     nodesCount = 1
        print(f"nodesCount: {nodesCount}")
        print(f"lastPTag {lastPTag}")
        parent_index = 0
        for CnodeID in range(nodesCount*2): # base 0. Children nodes being generated
            CnodeID += 1 # base 1 fix
            CnodeID += lastPTag # add right most parent id number 
            if CnodeID % 2 == 0:
                print(f"Cnode: {CnodeID}")
                # pair of children, associate with previous parent in respective box
                print(f"Parent Index: {parent_index}")
                # First find best splitting spot 
                # Identify correct parent node first
                print(f"firstPTag: {firstPTag}")
                parent_node = bottomLayer[parent_index]
                parentNodeInfo = ftree.get_node(parent_node.tag)
                # print(f"Parent Node Info: {parentNodeInfo.data.Bi[0]}")

                # Determine longest Axis
                axis = np.subtract(parentNodeInfo.data.Bi[0], parentNodeInfo.data.Bi[1])
                axis = np.argmax(abs(axis))

                # Determine starting K point for searching optimimal splitting point
                k = parentNodeInfo.data.Bi[0][axis]

                # Calculate splitting criteria and determine child node locations (Bi_l & Bi_l)
                (_fcost_min, k_split, Bi_l, Bi_r, Cl, Cr) = SAH(parentNodeInfo.data.Bi, parentNodeInfo.data.verts, k, axis) # fix Bo, should be Bi 

                # Capture intersected vertices in Bi_l and Bi_r somewhere
                (verts_l, verts_r) = vertsList(parentNodeInfo.data.verts, k_split, axis)

                # Add in two children nodes to parent
                ftree.create_node(CnodeID-1, CnodeID-1, parent=parent_node.tag, data=AABB(Bi_l, verts_l))
                ftree.create_node(CnodeID, CnodeID, parent=parent_node.tag, data=AABB(Bi_r, verts_r))

                #ftree.show()
                
                # Update parent index 
                parent_index += 1

    ftree.show()
    return ftree


def BiToCube(Bi):
    """
    Bi is two points defining the tree
    """
    #print(f"Bi: {Bi}")
    edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )
    verticies = (
    (Bi[1][0], Bi[0][1], Bi[0][2]),
    (Bi[1][0], Bi[1][1], Bi[0][2]),
    (Bi[0][0], Bi[1][1], Bi[0][2]),
    (Bi[0][0], Bi[0][1], Bi[0][2]),

    (Bi[1][0], Bi[0][1], Bi[1][2]),
    (Bi[1][0], Bi[1][1], Bi[1][2]),
    (Bi[0][0], Bi[0][1], Bi[1][2]),
    (Bi[0][0], Bi[1][1], Bi[1][2])

    )

    #print(verticies)

    # verticies = (
    # tuple(map(mul, (1, 1, -1) , (abs(scene_box[1][0]), abs(scene_box[0][1]), abs(scene_box[0][2])))),
    # tuple(map(mul, (1, 1, -1) , (abs(scene_box[1][0]), abs(scene_box[1][1]), abs(scene_box[0][2])))),
    # tuple(map(mul, (-1, 1, -1) , (abs(scene_box[0][0]), abs(scene_box[1][1]), abs(scene_box[0][2])))),
    # tuple(map(mul, (-1, 1, -1) , (abs(scene_box[0][0]), abs(scene_box[0][1]), abs(scene_box[0][2])))),
    
    # tuple(map(mul, (1, 1, 1) , (abs(scene_box[1][0]), abs(scene_box[0][1]), abs(scene_box[1][2])))),
    # tuple(map(mul, (1, 1, 1) , (abs(scene_box[1][0]), abs(scene_box[1][1]), abs(scene_box[1][2])))),
    # tuple(map(mul, (-1, 1, 1) , (abs(scene_box[0][0]), abs(scene_box[0][1]), abs(scene_box[1][2])))),
    # tuple(map(mul, (-1, 1, 1) , (abs(scene_box[0][0]), abs(scene_box[1][1]), abs(scene_box[1][2]))))
    # )


    glPushMatrix()
    glScalef(*scene_scale)
    glTranslatef(*scene_trans)

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])

    glEnd()
    glPopMatrix()

    return


    



def treeToCubes(tree):
    # Tree is ftree generated in BVHTree
    #print(tree.expand_tree())
    for node in tree.expand_tree(mode=Tree.DEPTH):
        pls = tree[node]
        #print(pls.tag)

        BiToCube(pls.data.Bi)
        
        #print(pls.data.Bi)




def main():
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 1, 500.0)
        glTranslatef(0.0, 0.0, -10)

        tree = BVHTree()
        #treeToCubes(tree)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        glTranslatef(-0.5,0,0)
                    if event.key == pygame.K_RIGHT:
                        glTranslatef(0.5,0,0)
                    if event.key == pygame.K_UP:
                        glTranslatef(0,1,0)
                    if event.key == pygame.K_DOWN:
                        glTranslatef(0,-1,0)
                        #glRotatef(1, 0, 1, 0)


            glRotatef(1, 5, 1, 0)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            Model()
            #Cube()
            treeToCubes(tree)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            pygame.display.flip()
            pygame.time.wait(10)
            # BVHTreeDict()
            # return

main()