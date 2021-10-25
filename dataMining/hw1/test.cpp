#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
using namespace std;

int main()
{

    int ctime;
    cin >> ctime;

    long long one, two, three;

    while (ctime--)
    {

        cin >> one >> two >> three;

        long long m = min(min(one, two), three);

        one -= m;
        two -= m;
        three -= m;

        long long sum = one + two * 2 + three * 3;

        if (sum % 2 == 0)
            cout << 0 << "\n";
        else
            cout << 1 << "\n";
    }

    return 0;
}
