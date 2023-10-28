#include "decoder_utils.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <memory>

using namespace std;

int main()
{  
    string unit_file = "/path/to/phones.txt";
    std::shared_ptr<asrdec::UnitTable> table = std::make_shared<asrdec::UnitTable>();

    table->build( unit_file );

    std::cout << "Table Size: " << table->size() << std::endl;

    for ( size_t i=0; i<table->size(); i++ )
    {
        std::cout << i << " : " << table->sym(i) << std::endl;
    }

    return 0;
}