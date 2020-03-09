# -*- coding:utf-8 -*-
# Author：Jason
# Reference: HanLP Project v1.x https://github.com/hankcs/HanLP
# Note: The Project refers to HanLP, so it follows the HanLP's License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct

class ByteArray:
    def __init__(self, _bytes):
        self._offset = 0
        self._bytes = _bytes
    
    def nextInt(self,):
        result = bytesHighFirstToInt(self._bytes, self._offset)
        self._offset += 4
        return result

    def nextFloat(self,):
        result = bytesHighFirstToFloat(self._bytes, self._offset)
        self._offset += 4
        return result

    def nextDouble(self,):
        result = bytesHighFirstToDouble(self._bytes, self._offset)
        self._offset += 8
        return result

    def nextBoolean(self,):
        return self.nextByte() == 1

    def nextByte(self,):
        result = self._bytes[self._offset]
        self._offset += 1
        return result

    def nextUnsignedShort(self,):
        a = self.nextByte()
        b = self.nextByte()
        return (((a & 0xff) << 8) | (b & 0xff))

    def nextUTF(self,):
        utflen = self.nextUnsignedShort()
        bytearr = b''
        chararr = ''
        count = 0

        for i in range(0, utflen):
            x = bytes([self.nextByte()])
            bytearr += x
        
        while count < utflen:
            c = int(bytearr[count]) & 0xff
            if c > 127: break;
            count += 1;
            chararr += chr(c)

        while count < utflen:
            c = int(bytearr[count]) & 0xff
            c = c >> 4
            
            if c == 7:
                count += 1
                chararr += chr(c)
            elif c == 13:
                count += 2
                char2 = int(bytearr[count - 1])
                chararr += chr((((c & 0x1F) << 6) | (char2 & 0x3F)))
            elif c == 14:
                count += 3;
                char2 = int(bytearr[count - 1])
                char3 = int(bytearr[count - 2])
                chararr += chr((((c & 0x0F) << 12) | ((char2 & 0x3F) << 6) | ((char3 & 0x3F) << 0)))
                
        return chararr

unpack_float = struct.Struct('f').unpack
unpack_double = struct.Struct('d').unpack
pack_ulong = struct.Struct('I').pack
pack_ulonglong = struct.Struct('Q').pack

def bytesHighFirstToInt(_bytes, _start):
    num = _bytes[_start + 3] & 0xFF
    num |= ((_bytes[_start + 2] << 8) & 0xFF00)
    num |= ((_bytes[_start + 1] << 16) & 0xFF0000)
    num |= ((_bytes[_start] << 24) & 0xFF000000)
    return num

def bytesHighFirstToFloat(_bytes, _start):
    l = bytesHighFirstToInt(_bytes, _start)
    #num = struct.unpack('f', struct.pack('L', l))[0]
    num = unpack_float(pack_ulong(l))[0]
    return num

def bytesHighFirstToDouble(_bytes, _start):
    l = ((_bytes[_start] << 56) & 0xFF00000000000000)
    l |= ((_bytes[1 + _start] << 48) & 0xFF000000000000)
    l |= ((_bytes[2 + _start] << 40) & 0xFF0000000000)
    l |= ((_bytes[3 + _start] << 32) & 0xFF00000000)
    l |= ((_bytes[4 + _start] << 24) & 0xFF000000)
    l |= ((_bytes[5 + _start] << 16) & 0xFF0000)
    l |= ((_bytes[6 + _start] << 8) & 0xFF00)
    l |= (_bytes[7 + _start] & 0xFF)
    #num = struct.unpack('d', struct.pack('Q', l))[0]
    num = unpack_double(pack_ulonglong(l))[0]
    return num
