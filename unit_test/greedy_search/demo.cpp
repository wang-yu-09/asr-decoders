#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

#include "greedy_search_decoder.hpp"

using namespace std;

int main()
{  
    asrdec::DecodeParams param;
    std::string work_dir = "/path/to/unit_test_data";
    param.unit_file = work_dir + "/phones.txt";
    param.blank_id = 0;

    asrdec::gsd::GreedyDecoder decoder;
    decoder.init( param );
    std::cout << "init decoder done!" << std::endl;

    float prob_mat[72*136];
    memset(prob_mat, 0x0, sizeof(prob_mat));

    std::ifstream ifile;
    ifile.open( work_dir + "/prob_mat.txt" );
    std::string line;
    int row_idx = 0;
    while (getline(ifile, line) )
    {
        std::vector<std::string> line_probs;
        asrdec::split_string( line, " ", line_probs );
        if ( line_probs.size() != 136 )
        {
            std::cout << "Wrong Prob Dim: " <<  line_probs.size() << std::endl;
            return -1;
        }
        for ( unsigned int j=0; j<line_probs.size(); j++ )
        {
            prob_mat[row_idx*136+j] = atof( line_probs[j].c_str() );
        }
        row_idx ++;
    }

    if ( row_idx != 72 )
    {
        std::cout << "Wrong Prob Frames: " <<  row_idx << std::endl;
        return -1;
    }
    std::cout << "read probability matrix done!" << std::endl;

    // if ( ! prob_mat.isContinuous() )
    // {   
    //     std::cerr << "Data address is not continuous! " << std::endl;
    //     return 0;
    // }

    //float * prob_in = (float *)prob_mat.data;
    asrdec::DecodeResult result;
    decoder.decode( prob_mat, 72, 136, result );
    std::cout << "decode done!" << std::endl;
    result.show();

    return 0;
}