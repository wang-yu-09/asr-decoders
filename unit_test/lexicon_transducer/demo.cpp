#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

#include "lexicon_transducer.hpp"

using namespace std;

int main()
{  
    asrdec::DecodeParams param;
    std::string work_dir = "/path/to/unit_test_data";
    // param.lexiconp_file = work_dir + "/lexiconp.txt";
    // param.unit_file = work_dir + "/phones.txt";
    param.lexicon_file = "/path/to/word_to_phoneme.txt";
    param.unit_file = "/path/to/phoneme.txt";
    param.unk_symbol = "<unk>";
    param.kenlm_file = "/path/to/word-2g.bin";
    param.prune_beam_size = 5;
    param.word_beam_size = 5;
    param.prefix_beam_size = 5;
    param.blank_id = 0;
    param.lm_weight = 0.1;

    asrdec::lt::LexiconTransducer decoder;
    decoder.init( param );
    std::cout << "init decoder done!" << std::endl;

    // std::string units = "_S T AA1 R T _R AH0 K ER0 D";
    std::vector<std::string> units_vec;
    asrdec::split_string(units, " ", units_vec);
    
    asrdec::DecodeResult result;
    decoder.decode(units_vec, result);

    for ( std::string & word : result.words )
    {
        std::cout << word << " ";
    }
    std::cout << "\n";

    return 0;
}