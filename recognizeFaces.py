#!/usr/bin/python
# Usage: python recognizeFaces.py <dirPath> <jsonInputFileName>

import os
import sys
import json
from collections import OrderedDict
import codecs
import cv2
import face_recognition
import numpy

from Params import Params



def loadDataFromJSON_asDict(jsonPath=''):
    # json_file = open(jsonInputFileName, 'r', encoding='utf-8')
    json_file = codecs.open(jsonInputFileName, 'r')

    dataJSON = json_file.read()
    # dataJSON = json_file.readlines()
    # dataJSON = ''.join(json_file.readlines())
    # dataJSON = json_file.read().encode('utf-8')

    data = json.loads(dataJSON)
    return data


def saveDataToJSON(data=OrderedDict(), jsonPath=''):
    with open(jsonPath, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)

        # print('\nDADOS GERADOS - ' + jsonOutputFileName)
        # print('    lista de alunos: ' + str(data['lista de alunos']))
        # print('    Faltas: ' + str(data['Faltas']))


def getAlunosNamesFromImageFileName(imgFiles=[]):
    alunosNames = []
    # print('getAlunosNamesFromImageFileName():')
    for file in imgFiles:
        name = file.split(' - ')[1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        if name.endswith('.JPG'):
            name = name.split('.JPG')[0]
        if name.endswith('.png'):
            name = name.split('.png')[0]
        if name.endswith('.PNG'):
            name = name.split('.PNG')[0]
        alunosNames.append(name)
        # print('    file:', file, '    name:', name)
    return alunosNames


def getImagesListFromDirectory(pathDir=''):
    imgFiles = []
    for file in os.listdir(pathDir):
        if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png') or file.endswith('.PNG'):
            imgFiles.append(file)
            # print('    file:', file)
    imgFiles.sort()
    return imgFiles


def loadImages(pathDir='', imgFiles=[]):
    imgsList = []
    for file in imgFiles:
        pathImgFile = pathDir + '/' + file
        imagem = face_recognition.load_image_file(pathImgFile)
        imgsList.append(imagem)
        # cv2.imshow('', imagem)
        # cv2.waitKey(0)
    return imgsList


def convertRGB2BGR_3channels(img):
    return img[..., ::-1]


def convertFaceLocationFromOpencv2FaceRecognition(faces_locations):
    '''
    :param faces_locations[0]: x
           faces_locations[1]: y
           faces_locations[2]: width
           faces_locations[3]: height
    :return: (top, right, bottom, left)
    '''
    new_faces_locations = []
    for face_location in faces_locations:
        new_faces_locations.append([face_location[1], face_location[0]+face_location[2], face_location[1]+face_location[3], face_location[0]])
    return new_faces_locations


def convertFaceLocationFromFaceRecognition2Opencv(all_faces_locations):
    '''
    :param faces_locations[0]: top
           faces_locations[1]: right
           faces_locations[2]: bottom
           faces_locations[3]: left
    :return: (x, y, width, height)
    '''
    new_all_faces_locations = []
    for face_location in all_faces_locations:
        new_faces_locations = []
        for one_face_location in face_location:
            new_faces_locations.append([one_face_location[3], one_face_location[0], one_face_location[1]-one_face_location[3], one_face_location[2]-one_face_location[0]])
        new_all_faces_locations.append(new_faces_locations)
    return new_all_faces_locations


def detectAndCropFaces(imgsList=[], method='face_recognition'):
    '''
    :param imgsList: lista de imagens ja carregadas para a memoria
    :param method: 'face_recognition' ou 'opencv'
    :return:
    '''
    faceLocationList = []
    croppedFacesList = []
    face_cascade = cv2.CascadeClassifier('/home/ifmt/pycharm-workspace/face_recognition_capface_module/haarcascade_frontalface_default.xml')

    for img in imgsList:
        if method == 'face_recognition':
            faces_locations = face_recognition.face_locations(img=img, number_of_times_to_upsample=1, model="hog")
        elif method == 'opencv':
            imgBGR = convertRGB2BGR_3channels(img)
            imgBGR_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_locations = face_cascade.detectMultiScale(imgBGR, 1.3, 5)
            faces_locations = convertFaceLocationFromOpencv2FaceRecognition(faces_locations)

            # scaledImgBGR = cv2.resize(imgBGR, (0, 0), fx=0.25, fy=0.25)
            # cv2.imshow('', scaledImgBGR)
            # cv2.waitKey(0)

        if len(faces_locations) > 0:  # Se detectou uma face na imagem
            facesImgList = []
            croppedImgList = []
            for face_location in faces_locations:
                facesImgList.append(face_location)
                croppedImg = img[face_location[0]:face_location[2], face_location[3]:face_location[1]]
                croppedImgList.append(croppedImg)

                # croppedImgBGR = convertRGB2BGR_3channels(croppedImg)
                # cv2.imshow('Found face', croppedImgBGR)
                # cv2.waitKey(0)

            faceLocationList.append(facesImgList)
            croppedFacesList.append(croppedImgList)

    return faceLocationList, croppedFacesList


def computeBaseImagesFaceDescriptor(croppedFacesList=[], params=Params()):
    faceDescriptors_nparray = numpy.zeros((len(croppedFacesList), 128), dtype='float64')

    # print('computeBaseImagesFaceDescriptor():')
    for i in range(0, len(croppedFacesList)):
        croppedFace = croppedFacesList[i]
        descriptor = face_recognition.face_encodings(croppedFace, num_jitters=params.num_jitters_FaceNet)
        if len(descriptor) > 0:
            faceDescriptors_nparray[i] = descriptor[0]
            # print('    faceDescriptors_nparray[i]:', faceDescriptors_nparray[i])
    return faceDescriptors_nparray


def computeTestImagesFaceDescriptor(croppedFacesList=[], params=Params()):
    qtdeFaces = [len(croppedFacesList[i]) for i in range(0, len(croppedFacesList))]
    faceDescriptors_list = [numpy.zeros((qtdeFaces[i], 128), dtype='float64') for i in range(0, len(croppedFacesList))]

    # print('computeTestImagesFaceDescriptor():')
    for j in range(0, len(croppedFacesList)):
        imgFacesList = croppedFacesList[j]

        for i in range(0, len(imgFacesList)):
            croppedFace = imgFacesList[i]
            descriptor = face_recognition.face_encodings(croppedFace, num_jitters=params.num_jitters_FaceNet)
            if len(descriptor) > 0:
                faceDescriptors_list[j][i] = descriptor[0]
                # print('    faceDescriptors_nparray[i]:', faceDescriptors_nparray[i])
    return faceDescriptors_list


def filterRepeatedFaces(faceDescriptors_list=[], croppedFacesList=[]):
    # filterTolerance = 0.6
    filterTolerance = 0.4
    filteredFaceDescriptors_ndarray = faceDescriptors_list[0]  # adiciona as faces da primeira lista na lista final
    filteredCroppedFaces_list = croppedFacesList[0]              # adiciona as faces da primeira lista na lista final

    for i in range(1, len(faceDescriptors_list)):
        oneFaceDescriptor_list = faceDescriptors_list[i]  # aponta para a proxima lista de descritores
        for j in range(0, len(oneFaceDescriptor_list)):
            oneFace = croppedFacesList[i][j]  # aponta para uma face da lista selecionada
            oneDescriptor = oneFaceDescriptor_list[j]  # aponta para um descritor da lista selecionada
            distances = face_recognition.face_distance(filteredFaceDescriptors_ndarray, oneDescriptor)
            minDistance = distances.min()
            matchIndex = distances.argmin()

            if minDistance > filterTolerance:  # se nao houver matching o descritor entra na lista final
                filteredFaceDescriptors_ndarray = numpy.append(filteredFaceDescriptors_ndarray, oneDescriptor.reshape((1, oneDescriptor.shape[0])), axis=0)
                filteredCroppedFaces_list.append(oneFace)
                # cv2.imshow('Face Nao-Repetida', oneFace)
                # cv2.waitKey(0)
            else:
                pass
                # print('Face Repetida -> distance:', distances[matchIndex])
                # cv2.imshow('Face Repetida 1', filteredCroppedFaces_list[matchIndex])
                # cv2.imshow('Face Repetida 2', oneFace)
                # cv2.waitKey(0)

    return filteredFaceDescriptors_ndarray, filteredCroppedFaces_list


def joinFilesIntoOneList(imgsList=[]):
    newImgsList = imgsList[0]
    for i in range(1, len(imgsList)):
        newImgsList.append(imgsList[i][0])
    return newImgsList


def saveTestImagesWithRectangleFaces(pathDirTestImages='', testImgFiles=[], testImgsList=[], testFaceLocationList=[], croppedFacesList=[], testDescriptorsMatchings_list=[], baseAlunosNames=[]):
    dirDetectedFaces = 'test_images_with_detected_faces'
    pathDirDetectedFaces = pathDirTestImages + '/' + dirDetectedFaces
    if not os.path.exists(pathDirDetectedFaces):
        os.makedirs(pathDirDetectedFaces)

    faceLocationList = convertFaceLocationFromFaceRecognition2Opencv(testFaceLocationList)

    for i in range(0, len(testImgFiles)):
        imgFile = testImgFiles[i]
        imgFile_withDetectedFaces = imgFile.split('.')[0] + '_withDetectedAndRecognizedFaces.png'
        pathImgFile_withDetectedFaces = pathDirDetectedFaces + '/' + imgFile_withDetectedFaces
        img = testImgsList[i].copy()
        oneFaceLocationList = faceLocationList[i]

        for j in range(0, len(oneFaceLocationList)):
            x, y, w, h = oneFaceLocationList[j]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

            x2, y2, w2, h2 = (x, y+h, w, 50)
            cv2.rectangle(img, (x2-2, y2), (x2+w2+2, y2+h2), (0, 255, 0), cv2.FILLED)

            completeName = baseAlunosNames[testDescriptorsMatchings_list[i][j][2]]
            name = completeName.split(' ')[0] + ' ' + completeName.split(' ')[-1][0] + '.'
            font = cv2.FONT_HERSHEY_DUPLEX
            fontSize = (x2+w2 - x2) / 170
            cv2.putText(img=img, text=name, org=(x2+2, y2+int(h2/2)), fontFace=font, fontScale=fontSize, color=(255, 255, 255), thickness=2)

        img = convertRGB2BGR_3channels(img)
        cv2.imwrite(pathImgFile_withDetectedFaces, img)

        scaledImg = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow('', scaledImg)
        cv2.waitKey(5000)
    cv2.destroyAllWindows()



def adjustTestFaceSizeAndBaseFaceSize(testFace=numpy.array([]), baseFace=numpy.array([])):
    interpol = cv2.INTER_CUBIC
    if testFace.shape[0] <= baseFace.shape[0]:
        # baseFace = cv2.resize(baseFace, (testFace.shape[1], testFace.shape[0]), interpolation=interpol)
        testFace = cv2.resize(testFace, (baseFace.shape[1], baseFace.shape[0]), interpolation=interpol)
    else:
        # testFace = cv2.resize(testFace, (baseFace.shape[1], baseFace.shape[0]), interpolation=interpol)
        baseFace = cv2.resize(baseFace, (testFace.shape[1], testFace.shape[0]), interpolation=interpol)
    return testFace, baseFace



def saveTestCroppedFacesWithBaseCroppedFaces(pathDirTestImages='', testImgFiles=[], testImgsList=[], testFaceLocationList=[], croppedFacesList=[], testDescriptorsMatchings_list=[], baseCroppedFacesList=[], baseAlunosNames=[]):
    dirDetectedFaces = 'test_faces_with_recognized_base_faces'
    pathDirDetectedFaces = pathDirTestImages + '/' + dirDetectedFaces
    if not os.path.exists(pathDirDetectedFaces):
        os.makedirs(pathDirDetectedFaces)

    # faceLocationList = convertFaceLocationFromFaceRecognition2Opencv(testFaceLocationList)

    for i in range(0, len(testImgFiles)):
        imgFileName = testImgFiles[i].split('.')[0]
        oneCroppedFacesList = croppedFacesList[i]

        for j in range(0, len(oneCroppedFacesList)):
            oneCroppedTestFace = oneCroppedFacesList[j]
            indexMatchedBaseFace = testDescriptorsMatchings_list[i][j][2]
            matchedCroppedBaseFace = baseCroppedFacesList[indexMatchedBaseFace]
            matchedAlunoName = baseAlunosNames[indexMatchedBaseFace]
            imgTestFaceWithRecognizedBaseFace = imgFileName + '_face=' + str(j) + '_recognized=' + matchedAlunoName + '.png'
            pathImgTestFaceWithRecognizedBaseFace = pathDirDetectedFaces + '/' + imgTestFaceWithRecognizedBaseFace

            oneCroppedTestFace_adjusted, matchedCroppedBaseFace_adjusted = adjustTestFaceSizeAndBaseFaceSize(oneCroppedTestFace, matchedCroppedBaseFace)

            joinedImgFaces = numpy.concatenate((oneCroppedTestFace_adjusted, matchedCroppedBaseFace_adjusted), axis=1)  # stack horizontally

            joinedImgFaces = convertRGB2BGR_3channels(joinedImgFaces)
            cv2.imwrite(pathImgTestFaceWithRecognizedBaseFace, joinedImgFaces)

            # cv2.imshow('joinedImgFaces', joinedImgFaces)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()



def showFilteredCroppedFaces(filteredCroppedFaces_lists=[]):
    for oneFilteredCroppedFace in filteredCroppedFaces_lists:
        oneFilteredCroppedFace = convertRGB2BGR_3channels(oneFilteredCroppedFace)
        cv2.imshow('Filtered cropped face', oneFilteredCroppedFace)
        cv2.waitKey(0)
    cv2.destroyAllWindows()



def saveCroppedBaseImageFaces(pathTurmaName='', imgFiles=[], croppedFacesList=[], params=Params()):
    dirCroppedImgFaces = params.dirCroppedImgBaseFaces
    pathDirCroppedImgFaces = pathTurmaName + '/' + dirCroppedImgFaces
    if not os.path.exists(pathDirCroppedImgFaces):
        os.makedirs(pathDirCroppedImgFaces)

    for i in range(0, len(imgFiles)):
        croppedImgName = imgFiles[i].split('.')[0] + '_croppedFace.png'
        pathCroppedImgName = pathDirCroppedImgFaces + '/' + croppedImgName
        croppedFace = croppedFacesList[i]
        croppedFace = convertRGB2BGR_3channels(croppedFace)
        cv2.imwrite(pathCroppedImgName, croppedFace)



def loadCroppedFaces(pathTurmaName='', imgFiles=[], params=Params()):
    dirCroppedImgFaces = params.dirCroppedImgBaseFaces
    pathDirCroppedImgFaces = pathTurmaName + '/' + dirCroppedImgFaces
    croppedFacesList = []

    if os.path.exists(pathDirCroppedImgFaces):
        for i in range(0, len(imgFiles)):
            croppedImgName = imgFiles[i].split('.')[0] + '_croppedFace.png'
            pathCroppedImgName = pathDirCroppedImgFaces + '/' + croppedImgName
            croppedImg = face_recognition.load_image_file(pathCroppedImgName)
            croppedFacesList.append(croppedImg)

            # cv2.imshow('', croppedImg)
            # cv2.waitKey(0)

    return croppedFacesList



def saveBaseFaceDescriptor(pathTurmaName='', imgFiles=[], faceDescriptors_ndarray=numpy.array([]), params=Params()):
    dirDescriptorsCroppedImgBaseFaces = params.dirDescriptorsCroppedImgBaseFaces
    pathDirDescriptorsCroppedImgBaseFaces = pathTurmaName + '/' + dirDescriptorsCroppedImgBaseFaces
    if not os.path.exists(pathDirDescriptorsCroppedImgBaseFaces):
        os.makedirs(pathDirDescriptorsCroppedImgBaseFaces)

    for i in range(0, len(imgFiles)):
        descriptorFaceFileName = imgFiles[i].split('.')[0] + '_faceDescriptors.csv'
        pathDescriptorFaceFileName = pathDirDescriptorsCroppedImgBaseFaces + '/' + descriptorFaceFileName
        faceDescriptor = faceDescriptors_ndarray[i]
        numpy.savetxt(pathDescriptorFaceFileName, faceDescriptor, delimiter=',')



def loadBaseFaceDescriptor(pathTurmaName='', imgFiles=[], params=Params()):
    dirDescriptorsCroppedImgBaseFaces = params.dirDescriptorsCroppedImgBaseFaces
    pathDirDescriptorsCroppedImgBaseFaces = pathTurmaName + '/' + dirDescriptorsCroppedImgBaseFaces
    faceDescriptors_ndarray = numpy.zeros((len(imgFiles), 128))

    if os.path.exists(pathDirDescriptorsCroppedImgBaseFaces):
        for i in range(0, len(imgFiles)):
            descriptorFaceFileName = imgFiles[i].split('.')[0] + '_faceDescriptors.csv'
            pathDescriptorFaceFileName = pathDirDescriptorsCroppedImgBaseFaces + '/' + descriptorFaceFileName
            faceDescriptor = numpy.loadtxt(pathDescriptorFaceFileName, delimiter=',')
            faceDescriptors_ndarray[i] = faceDescriptor

    return faceDescriptors_ndarray



def computeBaseImagesDescriptors(pathDir='', turmaName='', params=Params()):
    sys.stdout.write('computeBaseImagesDescriptors(): loading image files names... ')
    sys.stdout.flush()
    pathTurmaName = pathDir + '/' + turmaName
    imgFiles = getImagesListFromDirectory(pathTurmaName)
    sys.stdout.write(str(len(imgFiles)) + ' files\n')
    sys.stdout.flush()

    sys.stdout.write('computeBaseImagesDescriptors(): separating alunos names... ')
    sys.stdout.flush()
    alunosNames = getAlunosNamesFromImageFileName(imgFiles)
    sys.stdout.write(str(len(alunosNames)) + ' names\n')
    sys.stdout.flush()

    sys.stdout.write('computeBaseImagesDescriptors(): loading images from disk... ')
    sys.stdout.flush()
    imgsList = loadImages(pathTurmaName, imgFiles)
    sys.stdout.write(str(len(imgsList)) + ' loaded images\n')
    sys.stdout.flush()

    pathDirCroppedFaces = pathTurmaName + '/' + params.dirCroppedImgBaseFaces
    if params.loadPrecomputedBaseFaceDescriptor == True and os.path.exists(pathDirCroppedFaces):
        sys.stdout.write('computeBaseImagesDescriptors(): loading cropped faces... ')
        sys.stdout.flush()
        croppedFacesList = loadCroppedFaces(pathTurmaName, imgFiles, params)
        sys.stdout.write(str(len(croppedFacesList)) + ' cropped faces loaded\n')
        sys.stdout.flush()

    else:
        sys.stdout.write('computeBaseImagesDescriptors(): detecting and cropping faces... ')
        sys.stdout.flush()
        faceLocationList, croppedFacesList = detectAndCropFaces(imgsList)
        faceLocationList = joinFilesIntoOneList(faceLocationList)
        croppedFacesList = joinFilesIntoOneList(croppedFacesList)
        sys.stdout.write(str(len(croppedFacesList)) + ' detected faces\n')
        sys.stdout.flush()

        sys.stdout.write('computeBaseImagesDescriptors(): saving cropped image faces... ')
        sys.stdout.flush()
        saveCroppedBaseImageFaces(pathTurmaName, imgFiles, croppedFacesList, params)
        sys.stdout.write(str(len(croppedFacesList)) + ' faces saved\n')
        sys.stdout.flush()

    pathDirDescriptorsCroppedImgBaseFaces = pathTurmaName + '/' + params.dirDescriptorsCroppedImgBaseFaces
    if params.loadPrecomputedBaseFaceDescriptor == True and os.path.exists(pathDirDescriptorsCroppedImgBaseFaces):
        sys.stdout.write('computeBaseImagesDescriptors(): loading face descriptors... ')
        sys.stdout.flush()
        faceDescriptors_nparray = loadBaseFaceDescriptor(pathTurmaName, imgFiles, params)
        sys.stdout.write(str(faceDescriptors_nparray.shape[0]) + ' face descriptors loaded\n')
        sys.stdout.flush()
    else:
        sys.stdout.write('computeBaseImagesDescriptors(): computing face descriptors (num_jitters_FaceNet: ' + str(params.num_jitters_FaceNet) + ')... ')
        sys.stdout.flush()
        faceDescriptors_nparray = computeBaseImagesFaceDescriptor(croppedFacesList, params)
        sys.stdout.write(str(faceDescriptors_nparray.shape[0]) + ' descriptors\n')
        sys.stdout.flush()

        sys.stdout.write('computeBaseImagesDescriptors(): saving face descriptors... ')
        sys.stdout.flush()
        saveBaseFaceDescriptor(pathTurmaName, imgFiles, faceDescriptors_nparray, params)
        sys.stdout.write(str(faceDescriptors_nparray.shape[0]) + ' descriptors saved\n')
        sys.stdout.flush()

    return alunosNames, imgsList, croppedFacesList, faceDescriptors_nparray




def computeTestImagesDescriptors(pathDir='', imgFiles=[], params=Params()):
    sys.stdout.write('\ncomputeTestImagesDescriptors(): loading images from disk... ')
    sys.stdout.flush()
    imgsList = loadImages(pathDir, imgFiles)
    sys.stdout.write(str(len(imgsList)) + ' loaded images\n')
    sys.stdout.flush()

    sys.stdout.write('computeTestImagesDescriptors(): detecting and cropping faces... ')
    sys.stdout.flush()
    faceLocationList, croppedFacesList = detectAndCropFaces(imgsList, method='opencv')
    # faceLocationList, croppedFacesList = detectAndCropFaces(imgsList, method='face_recognition')
    qtdeFaces = sum([len(faceLocationList[i]) for i in range(0, len(faceLocationList))])
    sys.stdout.write(str(qtdeFaces) + ' detected faces in total\n')
    sys.stdout.flush()

    sys.stdout.write('computeTestImagesDescriptors(): computing face descriptors (num_jitters_FaceNet: ' + str(params.num_jitters_FaceNet) + ')... ')
    sys.stdout.flush()
    faceDescriptors_list = computeTestImagesFaceDescriptor(croppedFacesList, params)
    qtdeTotalDescriptors = sum([len(faceDescriptors_list[i]) for i in range(0, len(faceDescriptors_list))])
    sys.stdout.write(str(qtdeTotalDescriptors) + ' descriptors in total\n')
    sys.stdout.flush()

    return imgsList, faceLocationList, croppedFacesList, faceDescriptors_list





def realizarMatchings(baseFaceDescriptors_nparray=numpy.array([]), testFaceDescriptors_list=[]):
    recognizeTolerance = 0.6
    testDescriptorsMatchings_list = []

    for i in range(0, len(testFaceDescriptors_list)):
        testFaceDescriptor_ndarray = testFaceDescriptors_list[i]
        oneTestFaceDescriptorMatching_list = []

        for j in range(0, len(testFaceDescriptor_ndarray)):
            oneTestFaceDescriptor_ndarray = testFaceDescriptor_ndarray[j]

            distances = face_recognition.face_distance(baseFaceDescriptors_nparray, oneTestFaceDescriptor_ndarray)
            minDistance = distances.min()
            matchIndex = distances.argmin()

            if minDistance <= recognizeTolerance:
                oneTestFaceDescriptorMatching_list.append((i, j, matchIndex))
            else:
                oneTestFaceDescriptorMatching_list.append((i, j, -1))

        testDescriptorsMatchings_list.append(oneTestFaceDescriptorMatching_list)

    return testDescriptorsMatchings_list




def generatePresenceList(nomesBaseAlunos=[], testDescriptorsMatchings_list=[]):
    alunosPresences_list = []
    for alunoIndex in range(0, len(nomesBaseAlunos)):
        alunoIsPresent = False
        for i in range(0, len(testDescriptorsMatchings_list)):
            oneTestDescriptorsMatchings_list = testDescriptorsMatchings_list[i]
            for j in range(0, len(oneTestDescriptorsMatchings_list)):
                oneMatching = oneTestDescriptorsMatchings_list[j]
                if oneMatching[2] != -1 and oneMatching[2] == alunoIndex:
                    alunoIsPresent = True
        alunosPresences_list.append(alunoIsPresent)
        # print('Aluno: ' + str(nomesBaseAlunos[alunoIndex]) + "    isPresent: " + str(alunoIsPresent))

    return alunosPresences_list




def recognizeFaces(pathDirBaseImages='', turmaName='', pathDirTestImages='', testImgFiles=[], params=Params()):
    baseAlunosNames, baseImgsList, \
    baseCroppedFacesList, baseFaceDescriptors_nparray = computeBaseImagesDescriptors(pathDirBaseImages, turmaName, params)

    testImgsList, testFaceLocationList, testCroppedFacesList, testFaceDescriptors_list = computeTestImagesDescriptors(pathDirTestImages, testImgFiles, params)

    sys.stdout.write('\nrecognizeFaces(): computing matchings between test and base faces... ')
    sys.stdout.flush()
    testDescriptorsMatchings_list = realizarMatchings(baseFaceDescriptors_nparray, testFaceDescriptors_list)
    sys.stdout.write('\n')
    sys.stdout.flush()

    sys.stdout.write('recognizeFaces(): saving test images with rectangle faces... ')
    sys.stdout.flush()
    saveTestImagesWithRectangleFaces(pathDirTestImages, testImgFiles, testImgsList, testFaceLocationList, testCroppedFacesList, testDescriptorsMatchings_list, baseAlunosNames)
    sys.stdout.write(str(len(testImgFiles)) + ' images saved\n')
    sys.stdout.flush()

    sys.stdout.write('recognizeFaces(): saving cropped test faces with recognized base faces... ')
    sys.stdout.flush()
    saveTestCroppedFacesWithBaseCroppedFaces(pathDirTestImages, testImgFiles, testImgsList, testFaceLocationList, testCroppedFacesList, testDescriptorsMatchings_list, baseCroppedFacesList, baseAlunosNames)
    sys.stdout.write(str(len(testImgFiles)) + ' images saved\n')
    sys.stdout.flush()

    # TODO: verificar matchings antes de gerar a lista de presenca
    # sys.stdout.write('recognizeFaces(): filtering repeated face descriptors... ')
    # sys.stdout.flush()
    # filteredFaceDescriptors_ndarray, filteredCroppedFaces_list = filterRepeatedFaces(faceDescriptors_list, croppedFacesList)
    # sys.stdout.write(str(filteredFaceDescriptors_ndarray.shape[0]) + ' filtered descriptors\n')
    # sys.stdout.flush()
    #
    # showFilteredCroppedFaces(filteredCroppedFaces_list)

    sys.stdout.write('recognizeFaces(): generating presence list... ')
    sys.stdout.flush()
    alunosPresences_list = generatePresenceList(baseAlunosNames, testDescriptorsMatchings_list)
    indexWhereAlunosPresences_list_equalToTrue = [i for i, n in enumerate(alunosPresences_list) if n == True]
    sys.stdout.write(str(len(indexWhereAlunosPresences_list_equalToTrue)) + ' alunos present\n')
    sys.stdout.flush()

    return baseAlunosNames, alunosPresences_list




# MAIN ROUTINE
if __name__ == '__main__':

    pathRepository = os.path.dirname(os.path.realpath(__file__))

    if len(sys.argv) > 1:
        pathDirTestImages = sys.argv[1]
        jsonInputFileName = sys.argv[2]

    else:
        # pathDirTestImages = pathRepository + '/uploadsTeste/upload_08-11-2018_16h27m'
        # pathDirTestImages = pathRepository + '/uploadsTeste/upload_26-11-2018_07h00m'
        # pathDirTestImages = pathRepository + '/uploadsTeste/upload_27-11-2018_16h00m'
        pathDirTestImages = '/home/ifmt/CapfaceUploads/bernardo_11-2018_12-00-15'
        jsonInputFileName = 'inicial.json'


    print('pathDirTestImages: ' + pathDirTestImages)
    print('jsonInputFileName: ' + jsonInputFileName)

    params = Params()

    jsonInputFileName = pathDirTestImages + '/' + jsonInputFileName
    jsonOutputFileName = 'final.json'
    jsonOutputFileName = pathDirTestImages + '/' + jsonOutputFileName

    pathDirBaseImages = pathRepository + '/imagensBase'


    params.loadPrecomputedBaseFaceDescriptor = True
    # params.loadPrecomputedBaseFaceDescriptor = False


    # params.num_jitters_FaceNet = 1
    params.num_jitters_FaceNet = 3
    # params.num_jitters_FaceNet = 5
    # params.num_jitters_FaceNet = 10
    # params.num_jitters_FaceNet = 15
    # params.num_jitters_FaceNet = 20


    data = loadDataFromJSON_asDict(jsonInputFileName)
    # print('DADOS LIDOS - ' + jsonInputFileName)
    # print('    Disciplina: ' + data['Disciplina'])
    # print('    Codigo: ' + data['Codigo'])
    # print('    Professor: ' + data['Professor'])
    # print('    Turma: ' + data['Turma'])
    # print('    data da aula: ' + data['data da aula'])
    # print('    Horario de inicio: ' + data['Horario de inicio'])
    # print('    Horario de fim: ' + data['Horario de fim'])
    # print('    quantidade de aulas: ' + data['quantidade de aulas'])
    # print('    Pergunta: ' + data['Pergunta'])
    # print('    Bimestre: ' + data['Bimestre'])
    # print('    Conteudo: ' + data['Conteudo'])
    # print('    imgFiles: ' + str(data['imgFiles']))



    nomesAlunos, presencasAlunos = recognizeFaces(pathDirBaseImages, data['Turma'], pathDirTestImages, data['imgFiles'], params)

    faltasAlunos = []
    for i in range(len(presencasAlunos)):
        if presencasAlunos[i] == True:
            faltasAlunos.append("0")  # se o aluno foi reconhecido recebe 0 faltas
        else:
            faltasAlunos.append(data['quantidade de aulas'])


    newData = OrderedDict()
    newData['Disciplina'] = data['Disciplina']
    newData['Codigo'] = data['Codigo']
    newData['Professor'] = data['Professor']
    newData['Turma'] = data['Turma']
    newData['data da aula'] = data['data da aula']
    newData['Horario de inicio'] = data['Horario de inicio']
    newData['Horario de fim'] = data['Horario de fim']
    newData['quantidade de aulas'] = data['quantidade de aulas']
    newData['Pergunta'] = data['Pergunta']
    newData['Bimestre'] = data['Bimestre']
    newData['Conteudo'] = data['Conteudo']
    newData['imgFiles'] = data['imgFiles']

    # newData['lista de alunos'] = ["Aluno1 Fulano1", "Aluno2 Fulano2"]
    # newData['Faltas'] = ["0", "0"]
    newData['lista de alunos'] = nomesAlunos
    newData['Faltas'] = faltasAlunos


    saveDataToJSON(newData, jsonOutputFileName)
