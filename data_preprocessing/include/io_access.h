// Copyright 2019 srabiee@cs.umass.edu
// College of Information and Computer Sciences,
// University of Massachusetts Amherst
//
//
// This software is free: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License Version 3,
// as published by the Free Software Foundation.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// Version 3 in the file COPYING that came with this distribution.
// If not, see <http://www.gnu.org/licenses/>.
// ========================================================================
#ifndef IVOA_IO_ACCESS
#define IVOA_IO_ACCESS

#include <vector>
#include <algorithm>
#include <glog/logging.h>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <jsoncpp/json/json.h>
#include <boost/filesystem.hpp>


#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bitset>
#include <cstdio>


namespace IVOA {
// Helper function to remove all files and folders in a given path
void RemoveDirectory(const std::string& path);

// Helper function that creates the given directory if it already does not
// exist. Returns false upon failure.
bool CreateDirectory(const std::string& path);

// Helper function to write JSON object to file
bool WriteJsonToFile(const std::string& path,
                const std::string& file_name,
                const Json::Value& json_obj);

// Given a direcotry whose content is supposed to be files named as ID numbers 
// in the format %010d<suffix>, it will fill file_names_prefix with  
// the prefix numbers of all available files.
bool GetFileNamePrefixes(const std::string &directory,
                         std::vector<int> *file_names_prefix);

enum PFM_endianness { BIG, LITTLE, ERROR};

inline void readFromStream(FILE* file, std::string format, char* c) {
  if (fscanf(file, format.c_str(), c) == 0) {
    std::cerr << "Error reading from stream" << std::endl;
    exit(1);
  }
}

class PFM {
public:
  PFM();
  inline bool is_little_big_endianness_swap(){
    if (this->endianess == 0.f) {
      std::cerr << "this-> endianness is not assigned yet!\n";
      exit(0);
    }
    else {
      uint32_t endianness = 0xdeadbeef;
      //std::cout << "\n" << std::bitset<32>(endianness) << std::endl;
      unsigned char * temp = (unsigned char *)&endianness;
      //std::cout << std::bitset<8>(*temp) << std::endl;
      PFM_endianness endianType_ = ((*temp ^ 0xef) == 0 ?
          LITTLE : (*temp ^ 0xde) == 0 ? BIG : ERROR);
      // ".pfm" format file specifies that: 
      // positive scale means big endianess;
      // negative scale means little endianess.
      return  ((BIG == endianType_) && (this->endianess < 0.f))
          || ((LITTLE == endianType_) && (this->endianess > 0.f));
    }
  }

  template<typename T>
  T * read_pfm(const std::string & filename) {
    FILE * pFile;
    pFile = fopen(filename.c_str(), "rb");
    char c[100];
    if (pFile != NULL) {
      readFromStream(pFile, "%s", c);
      // strcmp() returns 0 if they are equal.
      if (!strcmp(c, "Pf")) {
        readFromStream(pFile, "%s", c);
        // atoi: ASCII to integer.
        // itoa: integer to ASCII.
        this->width = atoi(c);
        readFromStream(pFile, "%s", c);
        this->height = atoi(c);
        unsigned int length_ = this->width * this->height;
        readFromStream(pFile, "%s", c);
        this->endianess = atof(c);

        fseek(pFile, 0, SEEK_END);
        long lSize = ftell(pFile);
        long pos = lSize - this->width*this->height * sizeof(T);
        fseek(pFile, pos, SEEK_SET);

        T* img = new T[length_];
        //cout << "sizeof(T) = " << sizeof(T);
        if (fread(img, sizeof(T), length_, pFile) != length_) {
          std::cerr << "error reading image" << std::endl;
          exit(1);
        }
        fclose(pFile);

        /* The raster is a sequence of pixels, packed one after another,
          * with no delimiters of any kind. They are grouped by row,
          * with the pixels in each row ordered left to right and
          * the rows ordered bottom to top.
          */
        T* tbimg = (T *)malloc(length_ * sizeof(T));// top-to-bottom.
        //PFM SPEC image stored bottom -> top reversing image           

                        
        for (unsigned int i = 0; i < this->height; i++) {
            memcpy(&tbimg[(this->height - i - 1)*(this->width)],
                &img[(i*(this->width))],
                (this->width) * sizeof(T));
        }


        if (this->is_little_big_endianness_swap()){
          std::cout << "little-big endianness transformation is "
                        "needed.\n";
          // little-big endianness transformation is needed.
          union {
              T f;
              unsigned char u8[sizeof(T)];
          } source, dest;

          for (unsigned int i = 0; i < length_; ++i) {
            source.f = tbimg[i];
            for (unsigned int k = 0, s_T = sizeof(T); k < s_T; k++)
                dest.u8[k] = source.u8[s_T - k - 1];
            tbimg[i] = dest.f;
            //cout << dest.f << ", ";
          }
        }
        delete[] img;
        return tbimg;
      }
      else {
        std::cout << "Invalid magic number!"
            << " No Pf (meaning grayscale pfm) is missing!!\n";
        fclose(pFile);
        exit(0);
      }
    }
    else {
        std::cout << "Cannot open file " << filename
            << ", or it does not exist!\n";
        fclose(pFile);
        exit(0);
    }
  }

  template<typename T>
  void write_pfm(const std::string & filename, const T* imgbuffer, 
      const float & endianess_) {
    std::ofstream ofs(filename.c_str(), std::ifstream::binary);
    // ** 1) Identifier Line: The identifier line contains the characters 
    // "PF" or "Pf". PF means it's a color PFM. 
    // Pf means it's a grayscale PFM.
    // ** 2) Dimensions Line:
    // The dimensions line contains two positive decimal integers, 
    // separated by a blank. The first is the width of the image; 
    // the second is the height. Both are in pixels.
    // ** 3) Scale Factor / Endianness:
    // The Scale Factor / Endianness line is a queer line that jams 
    // endianness information into an otherwise sane description 
    // of a scale. The line consists of a nonzero decimal number, 
    // not necessarily an integer. If the number is negative, that 
    // means the PFM raster is little endian. Otherwise, it is big 
    // endian. The absolute value of the number is the scale 
    // factor for the image.
    // The scale factor tells the units of the samples in the raster.
    // You use somehow it along with some separately understood unit 
    // information to turn a sample value into something meaningful, 
    // such as watts per square meter.

    ofs << "Pf\n"
        << this->width << " " << this->height << "\n"
        << endianess_ << "\n";
    /* PFM raster:
      * The raster is a sequence of pixels, packed one after another,
      * with no delimiters of any kind. They are grouped by row,
      * with the pixels in each row ordered left to right and
      * the rows ordered bottom to top.
      * Each pixel consists of 1 or 3 samples, packed one after another,
      * with no delimiters of any kind. 1 sample for a grayscale PFM
      * and 3 for a color PFM (see the Identifier Line of the PFM header).
      * Each sample consists of 4 consecutive bytes. The bytes represent
      * a 32 bit string, in either big endian or little endian format,
      * as determined by the Scale Factor / Endianness line of the PFM
      * header. That string is an IEEE 32 bit floating point number code.
      * Since that's the same format that most CPUs and compiler use,
      * you can usually just make a program use the bytes directly
      * as a floating point number, after taking care of the
      * endianness variation.
      */
    int length_ = this->width*this->height;
    this->endianess = endianess_;
    T* tbimg = (T *)malloc(length_ * sizeof(T));
    // PFM SPEC image stored bottom -> top reversing image
    for (int i = 0; i < this->height; i++) {
        memcpy(&tbimg[(this->height - i - 1)*this->width],
            &imgbuffer[(i*this->width)],
            this->width * sizeof(T));
    }

    if (this->is_little_big_endianness_swap()) {
        std::cout << "little-big endianness transformation is needed.\n";
        // little-big endianness transformation is needed.
        union {
            T f;
            unsigned char u8[sizeof(T)];
        } source, dest;

        for (int i = 0; i < length_; ++i) {
            source.f = tbimg[i];
            for (size_t k = 0, s_T = sizeof(T); k < s_T; k++)
                dest.u8[k] = source.u8[s_T - k - 1];
            tbimg[i] = dest.f;
            //cout << dest.f << ", ";
        }
    }

    ofs.write((char *)tbimg, this->width*this->height * sizeof(T));
    ofs.close();
    free(tbimg);
  }

  inline float getEndianess(){return endianess;}
  inline int getHeight(void){return height;}
  inline int getWidth(void){return width;}
  inline void setHeight(const int & h){height = h;}
  inline void setWidth(const int & w){width = w;}
private:
  unsigned int height;
  unsigned int width;
  float endianess;
};

} // namespace IVOA

#endif // IVOA_IO_ACCESS



