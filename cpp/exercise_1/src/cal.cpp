#include <iostream>
// #include <cmath>
using namespace std;
 
void reference(){
    int var1 = 5;
    const int &var2 = var1;
    cout<<&var1<<endl;
    cout<<&var2<<endl;

    cout<<var1<<endl;
    cout<<var2<<endl;

    var1 = 10;

    cout<<var1<<endl;
    cout<<var2<<endl;
    // return;
}

void pointer(){
    int var1 = 3;
    
    int *var2 = &var1;
    cout<<&var1<<endl;
    cout<<&var2<<endl;

    cout<<var1<<endl;
    cout<<var2<<endl;
    cout<<*var2<<endl;


}


int main(){
    double money, years, rate, sum; //宣告浮點型變數
    pointer();
    for(int i=0; i<=10; i++){
        // cout<<&var2<<endl;

    }
    return 0;
}