import numpy as np
def KernalMeshGeneration(totalLengthX, totalLengthY, totalLengthZ, centerX, centerY, centerZ, gapX, gapY, gapZ, filename="mesh.fdneut"):
    """
    Calculates the kernel rectangular mesh for the Immersed Boundary Method based on total lengths and gap sizes.
    This function generates a mesh file formatted for FDNEUT and saves it to the specified filename.
    Args:
        totalLengthX (float): Total length of the grid along the X-axis.
        totalLengthY (float): Total length of the grid along the Y-axis.
        totalLengthZ (float): Total length of the grid along the Z-axis.
        gapX (float): The gap size along the X-axis.
        gapY (float): The gap size along the Y-axis.
        gapZ (float): The gap size along the Z-axis.
        centerX (float): Center X coordinate of the mesh.
        centerY (float): Center Y coordinate of the mesh.
        centerZ (float): Center Z coordinate of the mesh.
        filename (str): The name of the file to which the mesh data is written. Defaults to 'mesh.fdneut'.

    Returns:
        None: The mesh data is directly written to a file in FDNEUT format.

    Raises:
        IOError: If the file cannot be written.

    Example:
        >>> KernalMeshGeneration(10, 10, 10, 5, 5, 5, 0, 0, 0, "example_mesh.fdneut")
    """
    ###浸入边界方法中心网格生成器。生成FDNEUT格式网格。
    
    #计算单元数
    cellNumberX, cellNumberY, cellNumberZ = calculate_cell_numbers(totalLengthX, gapX, totalLengthY, gapY, totalLengthZ, gapZ)

    # 生成结构化网格
    MeshX = np.linspace(centerX-totalLengthX/2.0, centerX+totalLengthX/2.0, cellNumberX + 1)
    MeshY = np.linspace(centerY-totalLengthY/2.0, centerY+totalLengthY/2.0, cellNumberY + 1)
    MeshZ = np.linspace(centerZ-totalLengthZ/2.0, centerZ+totalLengthZ/2.0, cellNumberZ + 1)
    MeshX, MeshY, MeshZ = np.meshgrid(MeshX, MeshY, MeshZ, indexing='ij')

    # 准备有用的数据
    ##计算所有node数量
    numNodes = (cellNumberX + 1) * (cellNumberY + 1) * (cellNumberZ + 1)
    ##计算所有volume数量
    numVolume = cellNumberX * cellNumberY * cellNumberZ
    ##计算所有体元加面元数量
    numVolumePlusnumFace = numVolume + cellNumberX * cellNumberY * 2 + cellNumberX * cellNumberZ *2 + cellNumberY * cellNumberZ *2 # the volumn + 6 face
    ##6face + 1 volume
    numEltGroupsForKernel = 7
    ## 1 for volume 2-7 for different face x- x+ z+ z- y+ y-
    idGroup = 1
    ##体元与面元的编号
    element_id = 1
    ## 人为定义的边界名称Mesh Boundary layer name, will be incremented for each face
    FAMNumber = 1
    ##网格上对应的每个格点的编号
    MeshId = np.arange(1,numNodes+1).reshape(cellNumberX + 1, cellNumberY + 1, cellNumberZ + 1)
    # 遍历每个单元格
    dataTemple=[]
    for i in range(cellNumberX):
        for j in range(cellNumberY):
            for k in range(cellNumberZ):
                # 获取当前单元格的8个顶点编号
                # 遍历网格，假设i, j, k 不越界
                # 保存网格点索引信息，不包括 element_id
                element_data = [
                MeshId[i+1, j, k+1], MeshId[i+1, j, k], MeshId[i, j, k+1], MeshId[i, j, k],
                MeshId[i+1, j+1, k+1], MeshId[i+1, j+1, k], MeshId[i, j+1, k+1], MeshId[i, j+1, k]
                ]
                dataTemple.append(element_data)

    # 转换为 NumPy 数组，并指定整数类型
    volume_contain = np.array(dataTemple, dtype=np.int64)
    ##########处理面数据
    # 计算面上的四边形数量
    numberFaceX = cellNumberY * cellNumberZ
    numberFaceY = cellNumberX * cellNumberZ
    numberFaceZ = cellNumberX * cellNumberY

    # 创建一个列表，其中每个元素都是对应面的四边形数据
    faces_contain = [
    np.zeros((numberFaceX, 4), dtype=int),  # 面1 X-
    np.zeros((numberFaceX, 4), dtype=int),  # 面2 X+
    np.zeros((numberFaceZ, 4), dtype=int),  # 面3 Z+
    np.zeros((numberFaceZ, 4), dtype=int),  # 面4 Z-
    np.zeros((numberFaceY, 4), dtype=int),  # 面5 Y+
    np.zeros((numberFaceY, 4), dtype=int)   # 面6 Y-
    ]
    ###装填x-
    index = 0
    for j in range(cellNumberY):
        for k in range(cellNumberZ):
            faces_contain[0][index, :] = [
                MeshId[0, j, k+1], MeshId[0, j+1, k+1], MeshId[0, j+1, k], MeshId[0, j, k]
            ]
            index += 1
    ###装填x+
    index = 0
    for j in range(cellNumberY):
        for k in range(cellNumberZ):
            faces_contain[1][index, :] = [
                MeshId[cellNumberX, j+1, k+1], MeshId[cellNumberX, j, k+1], MeshId[cellNumberX, j, k], MeshId[cellNumberX, j+1, k]
            ]
            index += 1
    ### 装填 z+
    index = 0
    for i in range(cellNumberX):
        for j in range(cellNumberY):
            faces_contain[2][index, :] = [
                MeshId[i, j, cellNumberZ], MeshId[i+1, j, cellNumberZ], MeshId[i+1, j+1, cellNumberZ], MeshId[i, j+1, cellNumberZ]
            ]
            index += 1

    ### 装填 z-
    index = 0
    for i in range(cellNumberX):
        for j in range(cellNumberY):
            faces_contain[3][index, :] = [
                MeshId[i, j, 0], MeshId[i, j+1, 0], MeshId[i+1, j+1, 0], MeshId[i+1, j, 0]
            ]
            index += 1

    ### 装填 y+
    index = 0
    for i in range(cellNumberX):
        for k in range(cellNumberZ):
            faces_contain[4][index, :] = [
                MeshId[i, cellNumberY, k+1], MeshId[i+1, cellNumberY, k+1], MeshId[i+1, cellNumberY, k], MeshId[i, cellNumberY, k]
            ]
            index += 1

    ### 装填 y-
    index = 0
    for i in range(cellNumberX):
        for k in range(cellNumberZ):
            faces_contain[5][index, :] = [
                MeshId[i+1, 0, k+1], MeshId[i, 0, k+1], MeshId[i, 0, k], MeshId[i+1, 0, k]
            ]
            index += 1

    
    # Writing to FDNEUT format
    with open(filename, "w") as file:
        headerTotal=create_header(numNodes, numVolumePlusnumFace, numEltGroupsForKernel)
        file.write(headerTotal)
        for i in range(MeshX.shape[0]):
            for j in range(MeshY.shape[1]):
                for k in range(MeshZ.shape[2]):
                    file.write(f"{MeshId[i,j,k]:10}   {MeshX[i,j,k]:15.10e}   {MeshY[i,j,k]:15.10e}   {MeshZ[i,j,k]:15.10e}\n")
        headerVolume = create_volume_header(idGroup, numVolume)
        file.write(headerVolume)
        idGroup += 1
        for row in volume_contain:
        # 写入 element_id 和体积数据
            file.write(f"{element_id:8}" + ' ' + ' '.join(f"{id:7}" for id in row) + '\n')
            element_id += 1  # 递增 element_id

        # 写入每个面的信息
        for face_index, face in enumerate(faces_contain):
            headerFace = create_header_face(idGroup, face.shape[0], FAMNumber)
            file.write(headerFace)
            idGroup += 1
            FAMNumber += 1
            for quad in face:
                file.write(f"{element_id:8} {quad[0]:7} {quad[1]:7} {quad[2]:7} {quad[3]:7}\n")
                element_id += 1
                
def OuterMeshGeneration(outerLengthX, outerLengthY, outerLengthZ, centerOuterX, centerOuterY, centerOuterZ, innerLengthX, innerLengthY, innerLengthZ, centerInnerX, centerInnerY, centerInnerZ, gapX, gapY, gapZ, filename="meshOut.fdneut"):
    """
Generates an outer mesh grid for use in simulations requiring the Immersed Boundary Method, tailored for larger computational domains with a less dense mesh compared to the kernel mesh. This function specifically supports dual-layered mesh structures, with distinct parameters for inner and outer mesh dimensions and centers. The inner settings should correspond to the previously configured mesh.

    Args:
        outerLengthX (float): Total length of the outer mesh along the X-axis.
        outerLengthY (float): Total length of the outer mesh along the Y-axis.
        outerLengthZ (float): Total length of the outer mesh along the Z-axis.
        centerOuterX (float): Center X coordinate of the outer mesh.
        centerOuterY (float): Center Y coordinate of the outer mesh.
        centerOuterZ (float): Center Z coordinate of the outer mesh.
        innerLengthX (float): Length of the inner mesh along the X-axis.
        innerLengthY (float): Length of the inner mesh along the Y-axis.
        innerLengthZ (float): Length of the inner mesh along the Z-axis.
        centerInnerX (float): Center X coordinate of the inner mesh.
        centerInnerY (float): Center Y coordinate of the inner mesh.
        centerInnerZ (float): Center Z coordinate of the inner mesh.
        gapX (float): The gap size between nodes along the X-axis.
        gapY (float): The gap size between nodes along the Y-axis.
        gapZ (float): The gap size between nodes along the Z-axis.
        filename (str): The filename for the generated mesh file. Defaults to 'meshOut.fdneut'.

    Returns:
        None: The function writes the mesh data directly to a file specified by the filename in FDNEUT format.

    Raises:
        IOError: If the file cannot be written due to IO issues such as permission errors or disk space limitations.

    Example:
        >>> OuterMeshGeneration(
                10, 10, 10, 0, 0, 0,
                1, 1, 1, 0, 0, 0,
                0.1, 0.1, 0.1,
                "example_outer_mesh.fdneut")
    """
    #计算外网格单元数
    outerCellNumberX, outerCellNumberY, outerCellNumberZ = calculate_cell_numbers(outerLengthX, gapX, outerLengthY, gapY, outerLengthZ, gapZ)

    # 生成结构化外网格
    outerMeshX = np.linspace(centerOuterX-outerLengthX/2.0, centerOuterX+outerLengthX/2.0, outerCellNumberX + 1)
    outerMeshY = np.linspace(centerOuterY-outerLengthY/2.0, centerOuterY+outerLengthY/2.0, outerCellNumberY + 1)
    outerMeshZ = np.linspace(centerOuterZ-outerLengthZ/2.0, centerOuterZ+outerLengthZ/2.0, outerCellNumberZ + 1)
    outerMeshX, outerMeshY, outerMeshZ = np.meshgrid(outerMeshX, outerMeshY, outerMeshZ, indexing='ij')

    #计算内网格单元数
    innerCellNumberX, innerCellNumberY, innerCellNumberZ = calculate_cell_numbers(innerLengthX, gapX, innerLengthY, gapY, innerLengthZ, gapZ)

    #找到内部单元对应外部网格的起止位置
    indexStartX = find_exact_index(outerMeshX[:,0,0], centerInnerX-innerLengthX/2.0)
    indexEndX = find_exact_index(outerMeshX[:,0,0], centerInnerX+innerLengthX/2.0)
    indexStartY = find_exact_index(outerMeshY[0,:,0], centerInnerY-innerLengthY/2.0)
    indexEndY = find_exact_index(outerMeshY[0,:,0], centerInnerY+innerLengthY/2.0)
    indexStartZ = find_exact_index(outerMeshZ[0,0,:], centerInnerZ-innerLengthZ/2.0)
    indexEndZ = find_exact_index(outerMeshZ[0,0,:], centerInnerZ+innerLengthZ/2.0)

    ##网格上对应的每个格点的编号
    MeshId = np.zeros((outerCellNumberX + 1, outerCellNumberY + 1, outerCellNumberZ + 1), dtype=int)

    node_id = 1

    for i in range(MeshId.shape[0]):
        for j in range(MeshId.shape[1]):
            for k in range(MeshId.shape[2]):
            # 检查是否在指定的内部网格区域内
                if indexStartX < i < indexEndX and indexStartY < j < indexEndY and indexStartZ < k < indexEndZ:
                    MeshId[i, j, k] = -1
                else:
                    MeshId[i, j, k] = node_id
                    node_id += 1
    node_id -= 1

# 准备有用的数据
    ##计算所有node数量
    numNodes = node_id
    ##计算所有volume数量
    numVolume = outerCellNumberX * outerCellNumberY * outerCellNumberZ - innerCellNumberX * innerCellNumberY * innerCellNumberZ
    ##计算所有体元加面元数量
    numVolumePlusnumFace = numVolume + outerCellNumberX * outerCellNumberY * 2 + outerCellNumberX * outerCellNumberZ *2 + outerCellNumberY * outerCellNumberZ *2 + innerCellNumberX * innerCellNumberY * 2 + innerCellNumberX * innerCellNumberZ *2 + innerCellNumberY * innerCellNumberZ *2 #the volumn+6 face outer+6 face inner
    ##6face + 6face + 1 volume
    numEltGroupsForOuter = 13
    ## 1 for volume 2-7 for inner different face x- x+ z+ z- y+ y- 8-13 for outer different face x- x+ z+ z- y+ y-
    idGroup = 1
    ## 面元与体元编号
    element_id = 1
    ## 人为定义的边界名称Mesh Boundary layer name, will be incremented for each face
    FAMNumber = 1
    
    # 遍历每个单元格，保存体元信息
    dataTemple=[]
    for i in range(outerCellNumberX):
        for j in range(outerCellNumberY):
            for k in range(outerCellNumberZ):
            # 获取当前单元格的8个顶点编号
            # 遍历网格，假设i, j, k 不越界
            # 保存网格点索引信息，不包括 element_id
            #去除越界的结果
                element_data = [
                MeshId[i+1, j, k+1], MeshId[i+1, j, k], MeshId[i, j, k+1], MeshId[i, j, k],
                MeshId[i+1, j+1, k+1], MeshId[i+1, j+1, k], MeshId[i, j+1, k+1], MeshId[i, j+1, k]
                ]
                if -1 not in element_data:
                    dataTemple.append(element_data)

    # 转换为 NumPy 数组，并指定整数类型
    volume_contain = np.array(dataTemple, dtype=np.int64)   

    ##########处理面数据
    # # 计算面上的四边形数量
    numberInnerFaceX = innerCellNumberY * innerCellNumberZ
    numberInnerFaceY = innerCellNumberX * innerCellNumberZ
    numberInnerFaceZ = innerCellNumberX * innerCellNumberY
    numberOuterFaceX = outerCellNumberY * outerCellNumberZ
    numberOuterFaceY = outerCellNumberX * outerCellNumberZ
    numberOuterFaceZ = outerCellNumberX * outerCellNumberY

    # # 创建一个列表，其中每个元素都是对应面的四边形数据
    faces_contain = [
    np.zeros((numberInnerFaceX, 4), dtype=int),  # 内面1 X-
    np.zeros((numberInnerFaceX, 4), dtype=int),  # 内面2 X+
    np.zeros((numberInnerFaceZ, 4), dtype=int),  # 内面3 Z+
    np.zeros((numberInnerFaceZ, 4), dtype=int),  # 内面4 Z-
    np.zeros((numberInnerFaceY, 4), dtype=int),  # 内面5 Y+
    np.zeros((numberInnerFaceY, 4), dtype=int),  # 内面6 Y-
    np.zeros((numberOuterFaceX, 4), dtype=int),  # 外面7 X-
    np.zeros((numberOuterFaceX, 4), dtype=int),  # 外面8 X+
    np.zeros((numberOuterFaceZ, 4), dtype=int),  # 外面9 Z+
    np.zeros((numberOuterFaceZ, 4), dtype=int),  # 外面10 Z-
    np.zeros((numberOuterFaceY, 4), dtype=int),  # 外面11 Y+
    np.zeros((numberOuterFaceY, 4), dtype=int)   # 外面12 Y-
    ]
    ###装填内x-
    index = 0
    for j in range(indexStartY,indexEndY):
        for k in range(indexStartZ,indexEndZ):
            faces_contain[0][index, :] = [
                MeshId[indexStartX, j, k+1], MeshId[indexStartX, j, k], MeshId[indexStartX, j+1, k], MeshId[indexStartX, j+1, k+1]
            ]
            index += 1
    ###装填内x+
    index = 0
    for j in range(indexStartY,indexEndY):
        for k in range(indexStartZ,indexEndZ):
            faces_contain[1][index, :] = [
                MeshId[indexEndX, j+1, k+1], MeshId[indexEndX, j+1, k], MeshId[indexEndX, j, k], MeshId[indexEndX, j, k+1]
            ]
            index += 1

    ### 装填内z+
    index = 0
    for i in range(indexStartX,indexEndX):
        for j in range(indexStartY,indexEndY):
            faces_contain[2][index, :] = [
                MeshId[i, j, indexEndZ], MeshId[i, j+1, indexEndZ], MeshId[i+1, j+1, indexEndZ], MeshId[i+1, j, indexEndZ]
            ]
            index += 1

    ### 装填内z-
    index = 0
    for i in range(indexStartX,indexEndX):
        for j in range(indexStartY,indexEndY):
            faces_contain[3][index, :] = [
                MeshId[i, j, indexStartZ], MeshId[i+1, j, indexStartZ], MeshId[i+1, j+1, indexStartZ], MeshId[i, j+1, indexStartZ]
            ]
            index += 1

    ### 装填内y+
    index = 0
    for i in range(indexStartX,indexEndX):
        for k in range(indexStartZ,indexEndZ):
            faces_contain[4][index, :] = [
                MeshId[i, indexEndY, k+1], MeshId[i, indexEndY, k], MeshId[i+1, indexEndY, k], MeshId[i+1, indexEndY, k+1]
            ]
            index += 1

    ### 装填内y-
    index = 0
    for i in range(indexStartX,indexEndX):
        for k in range(indexStartZ,indexEndZ):
            faces_contain[5][index, :] = [
                MeshId[i+1, indexStartY, k+1], MeshId[i+1, indexStartY, k], MeshId[i, indexStartY, k], MeshId[i, indexStartY, k+1]
            ]
            index += 1

    ###装填外x-
    index = 0
    for j in range(outerCellNumberY):
        for k in range(outerCellNumberZ):
            faces_contain[6][index, :] = [
                MeshId[0, j, k+1], MeshId[0, j+1, k+1], MeshId[0, j+1, k], MeshId[0, j, k]
            ]
            index += 1
    ###装填外x+
    index = 0
    for j in range(outerCellNumberY):
        for k in range(outerCellNumberZ):
            faces_contain[7][index, :] = [
                MeshId[outerCellNumberX, j+1, k+1], MeshId[outerCellNumberX, j, k+1], MeshId[outerCellNumberX, j, k], MeshId[outerCellNumberX, j+1, k]
            ]
            index += 1
    ### 装填外z+
    index = 0
    for i in range(outerCellNumberX):
        for j in range(outerCellNumberY):
            faces_contain[8][index, :] = [
                MeshId[i, j, outerCellNumberZ], MeshId[i+1, j, outerCellNumberZ], MeshId[i+1, j+1, outerCellNumberZ], MeshId[i, j+1, outerCellNumberZ]
            ]
            index += 1

    ### 装填 外z-
    index = 0
    for i in range(outerCellNumberX):
        for j in range(outerCellNumberY):
            faces_contain[9][index, :] = [
                MeshId[i, j, 0], MeshId[i, j+1, 0], MeshId[i+1, j+1, 0], MeshId[i+1, j, 0]
            ]
            index += 1

    ### 装填外y+
    index = 0
    for i in range(outerCellNumberX):
        for k in range(outerCellNumberZ):
            faces_contain[10][index, :] = [
                MeshId[i, outerCellNumberY, k+1], MeshId[i+1, outerCellNumberY, k+1], MeshId[i+1, outerCellNumberY, k], MeshId[i, outerCellNumberY, k]
            ]
            index += 1

    ### 装填外y-
    index = 0
    for i in range(outerCellNumberX):
        for k in range(outerCellNumberZ):
            faces_contain[11][index, :] = [
                MeshId[i+1, 0, k+1], MeshId[i, 0, k+1], MeshId[i, 0, k], MeshId[i+1, 0, k]
            ]
            index += 1
    
    # Writing to FDNEUT format
    with open(filename, "w") as file:
        headerTotal=create_header(numNodes, numVolumePlusnumFace, numEltGroupsForOuter)
        file.write(headerTotal)
        for i in range(MeshId.shape[0]):
            for j in range(MeshId.shape[1]):
                for k in range(MeshId.shape[2]):
                    # 仅当 MeshId 大于 0 时才执行写入操作
                    if MeshId[i, j, k] > 0:
                        file.write(f"{MeshId[i,j,k]:10}   {outerMeshX[i,j,k]:15.10e}   {outerMeshY[i,j,k]:15.10e}   {outerMeshZ[i,j,k]:15.10e}\n")
        headerVolume = create_volume_header(idGroup, numVolume)
        file.write(headerVolume)
        idGroup += 1
        for row in volume_contain:
        # 写入 element_id 和体积数据
            file.write(f"{element_id:8}" + ' ' + ' '.join(f"{id:7}" for id in row) + '\n')
            element_id += 1  # 递增 element_id
        # 写入每个面的信息
        for face_index, face in enumerate(faces_contain):
            headerFace = create_header_face(idGroup, face.shape[0], FAMNumber)
            file.write(headerFace)
            idGroup += 1
            FAMNumber += 1
            for quad in face:
                file.write(f"{element_id:8} {quad[0]:7} {quad[1]:7} {quad[2]:7} {quad[3]:7}\n")
                element_id += 1
                
def find_exact_index(outerMesh, target):
    """
    找到与给定目标距离小于指定间隙的网格点索引。

    Args:
        outerMesh (np.array): 一维网格数组。
        target (float): 目标位置。

    Returns:
        int: 满足条件的网格点索引。
    """
    # 计算所有点与目标的差的绝对值
    differences = np.abs(outerMesh - target)
    print(outerMesh,target)
    # 找到差值小于间隙的索引，0.1是一个缩放指标，理论上应该为0，但是数值上不允许
    indices = np.where(differences < 1e-8)[0]
    
    # 检查找到的索引数量
    if len(indices) != 1:
        raise ValueError("找到的符合条件的网格点不唯一或不存在，请检查网格配置或间隙设置。")
    print("满足条件的坐标的索引是: {}".format(indices[0]))
    if indices[0] == 0 or indices[0] == outerMesh.shape[0]-1:
        raise ValueError("内部网格维度过大，超出了合法范围。请调整内部网格尺寸或位置。")
    return indices[0]
  
def calculate_cell_numbers(totalLengthX, gapX, totalLengthY, gapY, totalLengthZ, gapZ):
    # 计算每个维度的网格数量
    cellNumberX = int(totalLengthX / gapX)
    cellNumberY = int(totalLengthY / gapY)
    cellNumberZ = int(totalLengthZ / gapZ)

    # 检查维度是否可以被间隔整除的内部函数
    def check_divisibility(length, gap, cellNumber):
        if abs(length - gap * cellNumber) > 1e-8:
            raise ValueError(f"Dimension {length} is not divisible by gap {gap}.")

    # 对每个维度应用可除性检查
    check_divisibility(totalLengthX, gapX, cellNumberX)
    check_divisibility(totalLengthY, gapY, cellNumberY)
    check_divisibility(totalLengthZ, gapZ, cellNumberZ)

    return cellNumberX, cellNumberY, cellNumberZ

def create_header(numNodes, numVolumePlusnumFace, numEltGroups):
    headerTotal = f"""** FIDAP NEUTRAL FILE
default_id_XuZhaoyue
VERSION    8.6

   NO. OF NODES   NO. ELEMENTS NO. ELT GROUPS          NDFCD          NDFVL
   {numNodes:12}   {numVolumePlusnumFace:12} {numEltGroups:14}              3              3
   STEADY/TRANS     TURB. FLAG FREE SURF FLAG    COMPR. FLAG   RESULTS ONLY
              0              0              0              0              0
TEMPERATURE/SPECIES FLAGS
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
PRESSURE FLAGS - IDCTS, IPENY MPDF
         1         1         0
NODAL COORDINATES
"""
    return headerTotal
def create_volume_header(idGroup, numVolume):
    headerVolume = f"""BOUNDARY CONDITIONS
         0         0         0     0.0
ELEMENT GROUPS
GROUP:{idGroup:9} ELEMENTS:{numVolume:10} NODES:            8 GEOMETRY:    3 TYPE:   3
ENTITY NAME:   fluid
"""
    return headerVolume
def create_header_face(idGroup, numberFace, FAMNumber):
    headerFace = f"""GROUP:{idGroup:9} ELEMENTS:{numberFace:10} NODES:            4 GEOMETRY:    1 TYPE:  17
ENTITY NAME:   FAM{FAMNumber}
"""
    return headerFace


def MeshKernelPart(totalLengthX, totalLengthY, totalLengthZ, gapX, gapY, gapZ, centerX, centerY, centerZ):
    import numpy as np
    """
    It is an old version!!!!!!!!!!!!!!!!!!!!!!!! 
    It is seriously clarified. But we develop a new version in order to modify.
    Calculates the kernel rectangular mesh for the Immersed Boundary Method based on total lengths and gap sizes.
    The origin set at zero!
    This function generates a mesh file formatted for FDNEUT.

    Args:
        totalLengthX (float): Total length of the grid along the X-axis.
        totalLengthY (float): Total length of the grid along the Y-axis.
        totalLengthZ (float): Total length of the grid along the Z-axis.
        gapX (float): The gap size along the X-axis.
        gapY (float): The gap size along the Y-axis.
        gapZ (float): The gap size along the Z-axis.

    Returns:
        None: Writes the mesh data directly to a file in FDNEUT format. 
    """
    # Calculate the number of divisions which is the cell, needed for each dimension
    cellNumberX = int(totalLengthX / gapX)
    cellNumberY = int(totalLengthY / gapY)
    cellNumberZ = int(totalLengthZ / gapZ)

    # Function to check if dimensions are divisible by gap size
    def check_divisibility(length, gap, cellNumber):
        if abs(length-gap*cellNumber)>1e-8:
            raise ValueError(f"Dimension {length} is not divisible by gap {gap}.")

    # Apply the divisibility check
    check_divisibility(totalLengthX, gapX, cellNumberX)
    check_divisibility(totalLengthY, gapY, cellNumberY)
    check_divisibility(totalLengthZ, gapZ, cellNumberZ)

    # Generate a structured grid
    MeshX = np.linspace(centerX-totalLengthX/2.0, centerX+totalLengthX/2.0, cellNumberX + 1)
    MeshY = np.linspace(centerY-totalLengthY/2.0, centerY+totalLengthY/2.0, cellNumberY + 1)
    MeshZ = np.linspace(centerZ-totalLengthZ/2.0, centerZ+totalLengthZ/2.0, cellNumberZ + 1)
    MeshX, MeshY, MeshZ = np.meshgrid(MeshX, MeshY, MeshZ,indexing='ij')

    # Prepare data for output
    ##Calculate total number of nodes
    numNodes = (cellNumberX + 1) * (cellNumberY + 1) * (cellNumberZ + 1)
    ##Calculate total number of volumes
    numVolume = cellNumberX * cellNumberY * cellNumberZ
    ##Used for count in file
    numVolumePlusnumFace = numVolume + cellNumberX * cellNumberY * 2 + cellNumberX * cellNumberZ *2 + cellNumberY * cellNumberZ *2 # the volumn + 6 face
    ##6face + 1 volume
    numEltGroupsForKernel = 7
    ## 1 for volume 2-7 for different face
    numGroup = 1
    ## Mesh Boundary layer name, will be incremented for each face
    FAMNumber = 1
    ### ...
    numberFace = 0
    ##The 
    MeshId = np.arange(numNodes).reshape(cellNumberX + 1, cellNumberY + 1, cellNumberZ + 1)

    # Validate node size
    if numNodes >= 10**10:
        raise ValueError("Node ID exceeds the maximum width.")

#Head of the Whole file(just used once)
    headerTotal = f"""** FIDAP NEUTRAL FILE
default_id_XuZhaoyue
VERSION    8.6

   NO. OF NODES   NO. ELEMENTS NO. ELT GROUPS          NDFCD          NDFVL
   {numNodes:12}   {numVolumePlusnumFace:12} {numEltGroupsForKernel:14}              3              3
   STEADY/TRANS     TURB. FLAG FREE SURF FLAG    COMPR. FLAG   RESULTS ONLY
              0              0              0              0              0
TEMPERATURE/SPECIES FLAGS
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
PRESSURE FLAGS - IDCTS, IPENY MPDF
         1         1         0
NODAL COORDINATES
"""
#Head of the Volume
    headerVolume = f"""BOUNDARY CONDITIONS
         0         0         0     0.0
ELEMENT GROUPS
GROUP:{numGroup:9} ELEMENTS:{numVolume:10} NODES:            8 GEOMETRY:    3 TYPE:   3
ENTITY NAME:   fluid
"""

    #Head of Face
    def create_header_face(numGroup, numberFace, FAMNumber):
        headerFace = f"""GROUP:{numGroup:9} ELEMENTS:{numberFace:10} NODES:            4 GEOMETRY:    1 TYPE:  17
ENTITY NAME:   FAM{FAMNumber}
"""
        return headerFace

    # Writing to FDNEUT format
    with open("mesh.fdneut", "w") as file:
        file.write(headerTotal)
        node_id = 1
        for i in range(MeshX.shape[0]):
            for j in range(MeshY.shape[1]):
                for k in range(MeshZ.shape[2]):
                    file.write(f"{node_id:10}   {MeshX[i,j,k]:15.10e}   {MeshY[i,j,k]:15.10e}   {MeshZ[i,j,k]:15.10e}\n")
                    MeshId[i,j,k]=node_id
                    node_id += 1
        file.write(headerVolume)
        numGroup += 1
        element_id = 1
        # 遍历每个单元格
        for i in range(cellNumberX):
            for j in range(cellNumberY):
                for k in range(cellNumberZ):
                # 获取当前单元格的8个顶点编号
                    file.write(f"{element_id:8} {MeshId[i+1, j, k+1]:7} {MeshId[i+1, j, k]:7} {MeshId[i, j, k+1]:7} {MeshId[i, j, k]:7} {MeshId[i+1, j+1, k+1]:7} {MeshId[i+1, j+1, k]:7} {MeshId[i, j+1, k+1]:7} {MeshId[i, j+1, k]:7}\n")
                    element_id += 1
        # 对x-调用函数生成头部信息
        numberFace = cellNumberY * cellNumberZ
        headerFace = create_header_face(numGroup, numberFace, FAMNumber)
        file.write(headerFace)
        numGroup += 1
        FAMNumber +=1
        for j in range(cellNumberY):
            for k in range(cellNumberZ):
                file.write(f"{element_id:8} {MeshId[0, j, k+1]:7} {MeshId[0, j+1, k+1]:7} {MeshId[0, j+1, k]:7} {MeshId[0, j, k]:7}\n")
                element_id += 1
       # 对x+调用函数生成头部信息
        headerFace = create_header_face(numGroup, numberFace, FAMNumber)
        file.write(headerFace)
        numGroup += 1
        FAMNumber += 1
        for j in range(cellNumberY):
            for k in range(cellNumberZ):
                file.write(f"{element_id:8} {MeshId[cellNumberX, j+1, k+1]:7} {MeshId[cellNumberX, j, k+1]:7} {MeshId[cellNumberX, j, k]:7} {MeshId[cellNumberX, j+1, k]:7}\n")
                element_id += 1
        # 对z+调用函数生成头部信息
        numberFace = cellNumberX * cellNumberY
        headerFace = create_header_face(numGroup, numberFace, FAMNumber)
        file.write(headerFace)
        numGroup += 1
        FAMNumber +=1
        for i in range(cellNumberX):
            for j in range(cellNumberY):
                file.write(f"{element_id:8} {MeshId[i, j, cellNumberZ]:7} {MeshId[i+1, j, cellNumberZ]:7} {MeshId[i+1, j+1, cellNumberZ]:7} {MeshId[i, j+1, cellNumberZ]:7}\n")
                element_id += 1
        # 对z-调用函数生成头部信息
        headerFace = create_header_face(numGroup, numberFace, FAMNumber)
        file.write(headerFace)
        numGroup += 1
        FAMNumber +=1
        for i in range(cellNumberX):
            for j in range(cellNumberY):
                file.write(f"{element_id:8} {MeshId[i, j, 0]:7} {MeshId[i, j+1, 0]:7} {MeshId[i+1, j+1, 0]:7} {MeshId[i+1, j, 0]:7}\n")
                element_id += 1
        # 对y+调用函数生成头部信息
        numberFace = cellNumberX * cellNumberZ
        headerFace = create_header_face(numGroup, numberFace, FAMNumber)
        file.write(headerFace)
        numGroup += 1
        FAMNumber +=1
        for i in range(cellNumberX):
            for k in range(cellNumberZ):
                file.write(f"{element_id:8} {MeshId[i, cellNumberY, k+1]:7} {MeshId[i+1, cellNumberY, k+1]:7} {MeshId[i+1, cellNumberY, k]:7} {MeshId[i, cellNumberY, k]:7}\n")
                element_id += 1
        # 对y-调用函数生成头部信息
        headerFace = create_header_face(numGroup, numberFace, FAMNumber)
        file.write(headerFace)
        numGroup += 1
        FAMNumber +=1
        for i in range(cellNumberX):
            for k in range(cellNumberZ):
                file.write(f"{element_id:8} {MeshId[i+1, 0, k+1]:7} {MeshId[i, 0, k+1]:7} {MeshId[i, 0, k]:7} {MeshId[i+1, 0, k]:7}\n")
                element_id += 1
    print("FDNEUT file created successfully!")
#use for test    print(MeshId)
