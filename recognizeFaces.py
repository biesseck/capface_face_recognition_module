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
    cropedFacesList = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
            cropedImgList = []
            for face_location in faces_locations:
                facesImgList.append(face_location)
                cropedImg = img[face_location[0]:face_location[2], face_location[3]:face_location[1]]
                cropedImgList.append(cropedImg)

                # cropedImgBGR = convertRGB2BGR_3channels(cropedImg)
                # cv2.imshow('Found face', cropedImgBGR)
                # cv2.waitKey(0)

            faceLocationList.append(facesImgList)
            cropedFacesList.append(cropedImgList)

    return faceLocationList, cropedFacesList


def computeBaseImagesFaceDescriptor(cropedFacesList=[], num_jitters_FaceNet=1):
    faceDescriptors_nparray = numpy.zeros((len(cropedFacesList), 128), dtype='float64')

    # print('computeBaseImagesFaceDescriptor():')
    for i in range(0, len(cropedFacesList)):
        cropedFace = cropedFacesList[i]
        descriptor = face_recognition.face_encodings(cropedFace, num_jitters=num_jitters_FaceNet)
        if len(descriptor) > 0:
            faceDescriptors_nparray[i] = descriptor[0]
            # print('    faceDescriptors_nparray[i]:', faceDescriptors_nparray[i])
    return faceDescriptors_nparray


def computeTestImagesFaceDescriptor(cropedFacesList=[], num_jitters_FaceNet=1):
    qtdeFaces = [len(cropedFacesList[i]) for i in range(0, len(cropedFacesList))]
    faceDescriptors_list = [numpy.zeros((qtdeFaces[i], 128), dtype='float64') for i in range(0, len(cropedFacesList))]

    # print('computeTestImagesFaceDescriptor():')
    for j in range(0, len(cropedFacesList)):
        imgFacesList = cropedFacesList[j]

        for i in range(0, len(imgFacesList)):
            cropedFace = imgFacesList[i]
            descriptor = face_recognition.face_encodings(cropedFace, num_jitters=num_jitters_FaceNet)
            if len(descriptor) > 0:
                faceDescriptors_list[j][i] = descriptor[0]
                # print('    faceDescriptors_nparray[i]:', faceDescriptors_nparray[i])
    return faceDescriptors_list


def filterRepeatedFaces(faceDescriptors_list=[], cropedFacesList=[]):
    # filterTolerance = 0.6
    filterTolerance = 0.4
    filteredFaceDescriptors_ndarray = faceDescriptors_list[0]  # adiciona as faces da primeira lista na lista final
    filteredCropedFaces_list = cropedFacesList[0]              # adiciona as faces da primeira lista na lista final

    for i in range(1, len(faceDescriptors_list)):
        oneFaceDescriptor_list = faceDescriptors_list[i]  # aponta para a proxima lista de descritores
        for j in range(0, len(oneFaceDescriptor_list)):
            oneFace = cropedFacesList[i][j]  # aponta para uma face da lista selecionada
            oneDescriptor = oneFaceDescriptor_list[j]  # aponta para um descritor da lista selecionada
            distances = face_recognition.face_distance(filteredFaceDescriptors_ndarray, oneDescriptor)
            minDistance = distances.min()
            matchIndex = distances.argmin()

            if minDistance > filterTolerance:  # se nao houver matching o descritor entra na lista final
                filteredFaceDescriptors_ndarray = numpy.append(filteredFaceDescriptors_ndarray, oneDescriptor.reshape((1, oneDescriptor.shape[0])), axis=0)
                filteredCropedFaces_list.append(oneFace)
                # cv2.imshow('Face Nao-Repetida', oneFace)
                # cv2.waitKey(0)
            else:
                pass
                # print('Face Repetida -> distance:', distances[matchIndex])
                # cv2.imshow('Face Repetida 1', filteredCropedFaces_list[matchIndex])
                # cv2.imshow('Face Repetida 2', oneFace)
                # cv2.waitKey(0)

    return filteredFaceDescriptors_ndarray, filteredCropedFaces_list


def joinFilesIntoOneList(imgsList=[]):
    newImgsList = imgsList[0]
    for i in range(1, len(imgsList)):
        newImgsList.append(imgsList[i][0])
    return newImgsList


def saveTestImagesWithRectangleFaces(pathDirTestImages='', testImgFiles=[], testImgsList=[], testFaceLocationList=[], cropedFacesList=[], testDescriptorsMatchings_list=[], baseAlunosNames=[]):
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

        # scaledImg = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        # cv2.imshow('', scaledImg)
        # cv2.waitKey(0)
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



def saveTestCropedFacesWithBaseCropedFaces(pathDirTestImages='', testImgFiles=[], testImgsList=[], testFaceLocationList=[], cropedFacesList=[], testDescriptorsMatchings_list=[], baseCropedFacesList=[], baseAlunosNames=[]):
    dirDetectedFaces = 'test_faces_with_recognized_base_faces'
    pathDirDetectedFaces = pathDirTestImages + '/' + dirDetectedFaces
    if not os.path.exists(pathDirDetectedFaces):
        os.makedirs(pathDirDetectedFaces)

    # faceLocationList = convertFaceLocationFromFaceRecognition2Opencv(testFaceLocationList)

    for i in range(0, len(testImgFiles)):
        imgFileName = testImgFiles[i].split('.')[0]
        oneCropedFacesList = cropedFacesList[i]

        for j in range(0, len(oneCropedFacesList)):
            oneCropedTestFace = oneCropedFacesList[j]
            indexMatchedBaseFace = testDescriptorsMatchings_list[i][j][2]
            matchedCropedBaseFace = baseCropedFacesList[indexMatchedBaseFace]
            matchedAlunoName = baseAlunosNames[indexMatchedBaseFace]
            imgTestFaceWithRecognizedBaseFace = imgFileName + '_face=' + str(j) + '_recognized=' + matchedAlunoName + '.png'
            pathImgTestFaceWithRecognizedBaseFace = pathDirDetectedFaces + '/' + imgTestFaceWithRecognizedBaseFace

            oneCropedTestFace_adjusted, matchedCropedBaseFace_adjusted = adjustTestFaceSizeAndBaseFaceSize(oneCropedTestFace, matchedCropedBaseFace)

            joinedImgFaces = numpy.concatenate((oneCropedTestFace_adjusted, matchedCropedBaseFace_adjusted), axis=1)  # stack horizontally

            joinedImgFaces = convertRGB2BGR_3channels(joinedImgFaces)
            cv2.imwrite(pathImgTestFaceWithRecognizedBaseFace, joinedImgFaces)

            # cv2.imshow('joinedImgFaces', joinedImgFaces)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()



def showFilteredCropedFaces(filteredCropedFaces_lists=[]):
    for oneFilteredCropedFace in filteredCropedFaces_lists:
        oneFilteredCropedFace = convertRGB2BGR_3channels(oneFilteredCropedFace)
        cv2.imshow('Filtered croped face', oneFilteredCropedFace)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def computeBaseImagesDescriptors(pathDir='', turmaName='', num_jitters_FaceNet=1):
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

    sys.stdout.write('computeBaseImagesDescriptors(): detecting and cropping faces... ')
    sys.stdout.flush()
    faceLocationList, cropedFacesList = detectAndCropFaces(imgsList)
    faceLocationList = joinFilesIntoOneList(faceLocationList)
    cropedFacesList = joinFilesIntoOneList(cropedFacesList)
    sys.stdout.write(str(len(cropedFacesList)) + ' detected faces\n')
    sys.stdout.flush()

    sys.stdout.write('computeBaseImagesDescriptors(): computing face descriptors (num_jitters_FaceNet: ' + str(num_jitters_FaceNet) + ')... ')
    sys.stdout.flush()
    faceDescriptors_nparray = computeBaseImagesFaceDescriptor(cropedFacesList, num_jitters_FaceNet)
    sys.stdout.write(str(faceDescriptors_nparray.shape[0]) + ' descriptors\n')
    sys.stdout.flush()

    return alunosNames, imgsList, faceLocationList, cropedFacesList, faceDescriptors_nparray




def computeTestImagesDescriptors(pathDir='', imgFiles=[], num_jitters_FaceNet=1):
    sys.stdout.write('\ncomputeTestImagesDescriptors(): loading images from disk... ')
    sys.stdout.flush()
    imgsList = loadImages(pathDir, imgFiles)
    sys.stdout.write(str(len(imgsList)) + ' loaded images\n')
    sys.stdout.flush()

    sys.stdout.write('computeTestImagesDescriptors(): detecting and cropping faces... ')
    sys.stdout.flush()
    faceLocationList, cropedFacesList = detectAndCropFaces(imgsList, method='opencv')
    # faceLocationList, cropedFacesList = detectAndCropFaces(imgsList, method='face_recognition')
    qtdeFaces = sum([len(faceLocationList[i]) for i in range(0, len(faceLocationList))])
    sys.stdout.write(str(qtdeFaces) + ' detected faces in total\n')
    sys.stdout.flush()

    sys.stdout.write('computeTestImagesDescriptors(): computing face descriptors (num_jitters_FaceNet: ' + str(num_jitters_FaceNet) + ')... ')
    sys.stdout.flush()
    faceDescriptors_list = computeTestImagesFaceDescriptor(cropedFacesList, num_jitters_FaceNet)
    qtdeTotalDescriptors = sum([len(faceDescriptors_list[i]) for i in range(0, len(faceDescriptors_list))])
    sys.stdout.write(str(qtdeTotalDescriptors) + ' descriptors in total\n')
    sys.stdout.flush()

    return imgsList, faceLocationList, cropedFacesList, faceDescriptors_list





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




def recognizeFaces(pathDirBaseImages='', turmaName='', pathDirTestImages='', testImgFiles=[], num_jitters_FaceNet=1):
    baseAlunosNames, baseImgsList, baseFaceLocationList, \
    baseCropedFacesList, baseFaceDescriptors_nparray = computeBaseImagesDescriptors(pathDirBaseImages, turmaName, num_jitters_FaceNet)

    testImgsList, testFaceLocationList, testCropedFacesList, testFaceDescriptors_list = computeTestImagesDescriptors(pathDirTestImages, testImgFiles, num_jitters_FaceNet)

    sys.stdout.write('\nrecognizeFaces(): computing matchings between test and base faces... ')
    sys.stdout.flush()
    testDescriptorsMatchings_list = realizarMatchings(baseFaceDescriptors_nparray, testFaceDescriptors_list)
    sys.stdout.write('\n')
    sys.stdout.flush()

    sys.stdout.write('recognizeFaces(): saving test images with rectangle faces... ')
    sys.stdout.flush()
    saveTestImagesWithRectangleFaces(pathDirTestImages, testImgFiles, testImgsList, testFaceLocationList, testCropedFacesList, testDescriptorsMatchings_list, baseAlunosNames)
    sys.stdout.write(str(len(testImgFiles)) + ' images saved\n')
    sys.stdout.flush()

    sys.stdout.write('recognizeFaces(): saving croped test faces with recognized base faces... ')
    sys.stdout.flush()
    saveTestCropedFacesWithBaseCropedFaces(pathDirTestImages, testImgFiles, testImgsList, testFaceLocationList, testCropedFacesList, testDescriptorsMatchings_list, baseCropedFacesList, baseAlunosNames)
    sys.stdout.write(str(len(testImgFiles)) + ' images saved\n')
    sys.stdout.flush()

    # sys.stdout.write('recognizeFaces(): filtering repeated face descriptors... ')
    # sys.stdout.flush()
    # filteredFaceDescriptors_ndarray, filteredCropedFaces_list = filterRepeatedFaces(faceDescriptors_list, cropedFacesList)
    # sys.stdout.write(str(filteredFaceDescriptors_ndarray.shape[0]) + ' filtered descriptors\n')
    # sys.stdout.flush()
    #
    # showFilteredCropedFaces(filteredCropedFaces_list)


    # nomesAlunos = alunosNames
    # presencasAlunos = alunosPresences

    return nomesAlunos, presencasAlunos




# MAIN ROUTINE
if __name__ == '__main__':

    # username = sys.argv[1]
    # pathDiretorioImagens = sys.argv[1]

    pathRepository = os.path.dirname(os.path.realpath(__file__))

    pathDirTestImages = pathRepository + '/uploadsTeste/upload_08-11-2018_16h27m'
    # pathDirTestImages = sys.argv[1]

    jsonInputFileName = 'inicial.json'
    # jsonInputFileName = sys.argv[2]
    jsonInputFileName = pathDirTestImages + '/' + jsonInputFileName

    jsonOutputFileName = 'final.json'
    jsonOutputFileName = pathDirTestImages + '/' + jsonOutputFileName

    pathDirBaseImages = pathRepository + '/imagensBase'


    # num_jitters_FaceNet = 1
    num_jitters_FaceNet = 3
    # num_jitters_FaceNet = 5
    # num_jitters_FaceNet = 10
    # num_jitters_FaceNet = 15
    # num_jitters_FaceNet = 20


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



    nomesAlunos, presencasAlunos = recognizeFaces(pathDirBaseImages, data['Turma'], pathDirTestImages, data['imgFiles'], num_jitters_FaceNet)

    faltasAlunos = []
    for i in range(len(presencasAlunos)):
        if presencasAlunos[i] == True:
            faltasAlunos.append(0)  # se o aluno foi reconhecido recebe 0 faltas
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
