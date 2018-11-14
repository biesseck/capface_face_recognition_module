#!/usr/bin/python
# Usage: python recognizeFaces_test.py <>

import sys
import json
from collections import OrderedDict
import codecs


# MAIN ROUTINE
if __name__ == '__main__':
    
    print('0) Execucao iniciada;  ')

    # username = sys.argv[1]
    # pathDiretorioImagens = sys.argv[1]

    # dirPath = sys.argv[1]
    dirPath = '/home/bernardo/faceDetectionRecognition/CapFace/imagensTeste/upload_08-11-2018_16h27m'

    # jsonInputFileName = sys.argv[2]
    jsonInputFileName = 'inicial.json'
    jsonInputFileName = dirPath + '/' + jsonInputFileName

    # jsonOutputFileName = 'teste2.json'
    jsonOutputFileName = 'final.json'
    jsonOutputFileName = dirPath + '/' + jsonOutputFileName


    data = []
    newData = OrderedDict()


    # json_file = open(jsonInputFileName, 'r', encoding='utf-8')
    json_file = codecs.open(jsonInputFileName, 'r')
    print('1) Arquivo aberto;  ')
    
    dataJSON = json_file.read()
    # dataJSON = json_file.readlines()
    # dataJSON = ''.join(json_file.readlines())
    # dataJSON = json_file.read().encode('utf-8')

    print('1.5) dataJSON:', dataJSON)

    #dataJSON_dec = dataJSON.decode('utf-8')
    #print('1.6) dataJSON_dec:', dataJSON_dec)

    data = json.loads(dataJSON)
    print('2) Dados carregados;  ')
    print('data: ' + str(data) + '; ')

    disciplina = data['Disciplina']
    
    try:
        codigo = data['Codigo']
    except:
        codigo = data['Código']   

    professor = data['Professor']
    turma = data['Turma']
    dataDaAula = data['data da aula']
    horaInicioAula = data['Horario de inicio']
    horaFimAula = data['Horario de fim']
    quantidadeAulas = data['quantidade de aulas']
    pergunta = data['Pergunta']
    bimestre = data['Bimestre']
    conteudoAula = data['Conteudo']

    print('DADOS LIDOS - ' + jsonInputFileName)
    print('    Disciplina: ' + disciplina)
    print('    Codigo: ' + codigo)
    print('    Professor: ' + professor)
    print('    Turma: ' + turma)
    print('    data da aula: ' + dataDaAula)
    print('    Horario de inicio: ' + horaInicioAula)
    print('    Horario de fim: ' + horaFimAula)
    print('    quantidade de aulas: ' + quantidadeAulas)
    print('    Pergunta: ' + pergunta)
    print('    Bimestre: ' + bimestre)
    print('    Conteudo: ' + conteudoAula)


    newData['Disciplina'] = data['Disciplina']
    
    try:    
        newData['Codigo'] = data['Codigo']
    except:
        newData['Código'] = data['Código']

    newData['Professor'] = data['Professor']
    newData['Turma'] = data['Turma']
    newData['data da aula'] = data['data da aula']
    newData['Horario de inicio'] = data['Horario de inicio']
    newData['Horario de fim'] = data['Horario de fim']
    newData['quantidade de aulas'] = data['quantidade de aulas']
    newData['Pergunta'] = data['Pergunta']
    newData['Bimestre'] = data['Bimestre']
    newData['Conteudo'] = data['Conteudo']
    newData['lista de alunos'] = ["Aluno1 Fulano1", "Aluno2 Fulano2"]
    newData['Faltas'] = ["0", "0"]


    with open(jsonOutputFileName, 'w', encoding='utf-8') as outfile:
        json.dump(newData, outfile, indent=2, ensure_ascii=False)

        print('\nDADOS GERADOS - ' + jsonOutputFileName)
        print('    lista de alunos: ' + str(newData['lista de alunos']))
        print('    Faltas: ' + str(newData['Faltas']))


    exit()








    with open(jsonInputFileName, 'r', encoding='utf-8') as json_file:
        print('1) Arquivo aberto;  ')
        data = json.load(json_file)
        
        print('2) Dados carregados;  ')
        print('data: ' + str(data) + '; ')

        # print('data: ', data, '\n')

        disciplina = data['Disciplina']
        codigo = data['Codigo']
        professor = data['Professor']
        turma = data['Turma']
        dataDaAula = data['data da aula']
        horaInicioAula = data['Horario de inicio']
        horaFimAula = data['Horario de fim']
        quantidadeAulas = data['quantidade de aulas']
        pergunta = data['Pergunta']
        bimestre = data['Bimestre']
        conteudoAula = data['Conteudo']

        print('DADOS LIDOS - ' + jsonInputFileName)
        print('    Disciplina' + disciplina)
        print('    Codigo' + codigo)
        print('    Professor' + professor)
        print('    Turma' + turma)
        print('    data da aula' + dataDaAula)
        print('    Horario de inicio' + horaInicioAula)
        print('    Horario de fim' + horaFimAula)
        print('    quantidade de aulas' + quantidadeAulas)
        print('    Pergunta' + pergunta)
        print('    Bimestre' + bimestre)
        print('    Conteudo' + conteudoAula)


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
        newData['lista de alunos'] = ["Aluno1 Fulano1", "Aluno2 Fulano2"]
        newData['Faltas'] = ["0", "0"]


    with open(jsonOutputFileName, 'w', encoding='utf-8') as outfile:
        json.dump(newData, outfile, indent=2, ensure_ascii=False)

        print('\nDADOS GERADOS - ' + jsonOutputFileName)
        print('    lista de alunos' + str(newData['lista de alunos']))
        print('    Faltas' + str(newData['Faltas']))
