#!/bin/bash
# Authors:
#   (1) Wang Yu, Oct 28, 2023

import numpy as np
import struct

def read_one_matrix_record(fp):
    '''
    Read an utterance.
    '''
    utt = ''
    while True:
        char = fp.read(1).decode()
        if (char == '') or (char == ' '):break
        utt += char
    utt = utt.strip()
    if utt == '':
        if fp.read() == b'':
            return (None,None,None,None,None,None)
        else:
            fp.close()
            print( "Miss utterance ID before utterance." )
            raise RuntimeError
    binarySymbol = fp.read(2).decode()
    if binarySymbol == '\0B':
        sizeSymbol = fp.read(1).decode()
        if sizeSymbol not in ["C","F","D"]:
            fp.close()
            if sizeSymbol == '\4':
                print( "This is not a matrix. May be a vector?" )
                raise RuntimeError
            else:
                print( "This might not be Kaldi archive data?" )
                raise RuntimeError

        dataType = sizeSymbol + fp.read(2).decode() 
        if dataType == 'CM ':
            fp.close()
            print( "This is compressed binary data. Please use kaldi command to decompress it firstly, such as copy-feat or may be others." )
            raise RuntimeError
        elif dataType == 'FM ':
            sampleSize = 4
        elif dataType == 'DM ':
            sampleSize = 8
        else:
            fp.close()
            print( f"Expected archive data type is one of FM(float32), DM(float64), CM(compressed data) but got {dataType}." )
            raise RuntimeError

        s1,rows,s2,cols = np.frombuffer(fp.read(10),dtype="int8,int32,int8,int32",count=1)[0]
        rows = int(rows)
        cols = int(cols)
        bufSize = rows * cols * sampleSize
        buf = fp.read(bufSize)
    else:
        fp.close()
        print( "Miss binary symbol before utterance. Is this a kaldi binary archive table?" )
        raise RuntimeError

    return (utt, dataType, rows, cols, bufSize, buf)

def read_one_vector_record(fp):
    '''
    Read an utterance.
    '''
    utt = ''
    while True:
        char = fp.read(1).decode()
        if (char == '') or (char == ' '):break
        utt += char
    utt = utt.strip()
    if utt == '':
        if fp.read() == b'':
            return (None,None,None,None,None)
        else:
            fp.close()
            print( "Miss utterance ID before utterance." )
            raise RuntimeError

    binarySymbol = fp.read(2).decode()
    if binarySymbol == '\0B':
        dataSize = fp.read(1).decode()
        if dataSize != '\4':
            fp.close()
            if dataSize not in ["C","F","D"]:
                print( "This is not a vector table. May be a matrix?" )
                raise RuntimeError
            else:
                print( f"We only support read size 4 int vector but got {dataSize}." )
                raise RuntimeError

        frames = int(np.frombuffer(fp.read(4),dtype='int32',count=1)[0])
        if frames == 0:
            buf = b""
        else:
            bufferSize = frames * 5
            buf = fp.read(bufferSize)
    else:
        fp.close()
        print( "Miss binary symbol before utterance. Is this a kaldi binary archive table?" )
        raise RuntimeError
    
    return (utt, 4, frames, bufferSize, buf)

def read_matrix_ark( matrix_ark_table ):
    '''
    Read data from kaldi binary archive table
    '''
    results = {}
    with open(matrix_ark_table,"rb") as fp:
        while True:
            (utt, dataType, rows, cols, bufSize, buf) = read_one_matrix_record(fp)
            if utt is None:
                break
            try:
                if dataType == 'FM ': 
                    newMatrix = np.frombuffer(buf, dtype=np.float32)
                else:
                    newMatrix = np.frombuffer(buf, dtype=np.float64)
            except Exception as e:
                e.args = ( f"Wrong matrix data format at utterance {utt}." + "\n" + e.args[0],)
                raise e	
            else:
                results[utt] = np.reshape(newMatrix,(rows,cols))

    return results

def read_vector_ark( vector_ark_table ):
    '''
    Read data from kaldi binary archive table
    '''
    results = {}
    with open(vector_ark_table,"rb") as fp:
        while True:
            (utt,dataSize,frames,bufSize,buf) = read_one_vector_record(fp)
            if utt is None:
                break
            vector = np.frombuffer(buf,dtype=[("size","int8"),("value","int32")],count=frames)
            vector = vector[:]["value"]
            results[utt] = vector

    return results

def write_matrix_ark( source, matrix_ark_table ):

    sorted_keys = sorted( source.keys() )

    with open(matrix_ark_table,"wb") as fw:

        for utt in sorted_keys:
            matrix = source[utt]

            assert len( matrix.shape ) == 2, "Matrix data is expected."

            data = ( utt+' ').encode()
            data += '\0B'.encode()
            if matrix.dtype == 'float32':
                data += 'FM '.encode()
            elif matrix.dtype == 'float64':
                data += 'DM '.encode()
            else:
                print( f'Expected "float32" or "float64" data, but got {matrix.dtype}.' )
                raise RuntimeError
            
            data += '\04'.encode()
            data += struct.pack(np.dtype('uint32').char,matrix.shape[0])
            data += '\04'.encode()
            data += struct.pack(np.dtype('uint32').char,matrix.shape[1])
            data += matrix.tobytes()

            fw.write( data )
            # print( "write done:", utt  )

def write_vector_ark( source, vector_ark_table ):

    sorted_keys = sorted( source.keys() )

    with open(vector_ark_table,"wb") as fw:

        for utt in sorted_keys:
            vector = source[utt]

            assert len(vector.shape) == 1, "Vector data is expected."
            assert len(vector.dtype) == "int32", "int32 vector is expected."

            oneRecord = []
            oneRecord.append( ( utt + ' ' + '\0B' + '\4' ).encode() )
            oneRecord.append( struct.pack(np.dtype('int32').char,vector.shape[0]) ) 
            for v in vector:
                oneRecord.append( '\4'.encode() + struct.pack(np.dtype('int32').char,v) )
            oneRecord = b"".join(oneRecord)

            fw.write(oneRecord)
