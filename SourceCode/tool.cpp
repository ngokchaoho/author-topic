#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

typedef std::pair<int, int> key;
//Functor for comparing pair<int,int>
struct myeq{
  bool operator() (const std::pair<int, int> & x, const std::pair<int, int> & y) const{
    return x.first == y.first && x.second == y.second;
  }
};
//Functor for Hash function
struct myhash{
private:
  const std::hash<int> h_int;
public:
  myhash() : h_int() {}
  size_t operator()(const std::pair<int, int> & p) const{
    size_t seed = h_int(p.first);
    return h_int(p.second) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }
};
//Function for splitting string
std::vector<std::string> split_string(std::string s, std::string c){
  std::vector<std::string> ret;
  for(int i = 0, n = 0; i <= s.length(); i = n + 1){
    n = s.find_first_of(c, i);
    if(n == std::string::npos) n = s.length();
    std::string tmp = s.substr(i, n-i);
    ret.push_back(tmp);
  }
  return ret;
}
//pdf of uniformly distributed random number
double uniform_rand(){
  return (double)rand() * (1.0 / (RAND_MAX + 1.0));
}
