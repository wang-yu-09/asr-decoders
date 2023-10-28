#include "decoder_utils.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <memory>

using namespace std;

int main()
{  
    string unit_file = "/path/to/phoneme.txt";
    string lexicon = "/path/to/word_to_phoneme.txt";
    std::shared_ptr<asrdec::UnitTable> utable = std::make_shared<asrdec::UnitTable>();

    if ( utable->build( unit_file ) < 0 )
    {
        std::cout << "build unit table filed!\n";
        return 0;
    }
    std::cout << utable->size() << "\n";
    asrdec::PruneTable ptable;
    ptable.build( lexicon, utable );

    ptable.save("/path/to/table.txt");

    return 0;
}