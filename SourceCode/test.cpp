#include "tool.cpp"
#include "iostream"
using namespace std;
int main(){
	myeq instance;
	pair<int,int> yi;
	pair<int,int> er;
	yi.first=1;
	yi.second=1;
	er.first=1;
	er.second=0;
	cout<<instance(yi,er);
}