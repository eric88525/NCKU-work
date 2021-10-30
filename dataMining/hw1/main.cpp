#include "fpGrowth.h"
#include "apriori.h"

// get all transations
vector<vector<string>> readIBMData(string filePath)
{
    string id, tranID, item;

    ifstream infile(filePath);

    map<string, vector<string>> datas;

    while (infile >> id >> tranID >> item)
    {
        datas[tranID].push_back(item);
    }

    vector<vector<string>> result;

    for (auto &d : datas)
    {
        result.push_back(d.second);
    }
    return result;
}

vector<vector<string>> readKaggleData(string filepath)
{

    string inp;
    ifstream infile(filepath);

    map<string, vector<string>> datas;

    while (getline(infile, inp))
    {
        datas[inp.substr(0, inp.find(','))].push_back(inp.substr(inp.find(',') + 1));
    }

    vector<vector<string>> result;
    for (auto &d : datas)
    {
        result.push_back(d.second);
    }

    return result;
}

// print result to file
void printResult(const vector<assoInfo> &freqSet, string setPath, string rulePath, float minSupport, float minConfidence, float transCount)
{
    ofstream myfile;
    myfile.open(setPath);

    int totalRules = 0;

    map<string, float> ruleCount;

    ofstream rulefile;
    rulefile.open(rulePath);

    // create itemset:freq map
    rulefile << "count | itemSet\n";
    for (const auto &fSet : freqSet)
    {
        string k = "";
        for (const auto &item : fSet.itemSet)
            k += item + ", ";

        ruleCount[k] = fSet.support;

        rulefile << setw(5) << setprecision(4) << fSet.support << " | { " << k << "}\n";
    }
    rulefile.close();

    myfile << " Sup  | Conf  | Rule\n";
    myfile << "====================\n";

    //  confidence filter & print reuslt to file
    for (const auto &itSet : freqSet)
    {
        auto allCombs = allCombination(itSet.itemSet);

        for (const auto &comb : allCombs)
        {

            float sup = float(itSet.support) / transCount;
            float confi = float(itSet.support) / ruleCount[comb.first];

            if (confi > 1)
                cout << "wrong\n";
            if (confi < minConfidence)
                continue;

            myfile << setprecision(5) << setw(5) << sup << " | " << setprecision(5) << setw(5) << confi << " | ";
            myfile << "{ " << comb.first << "} => { " << comb.second << "}\n";

            totalRules++;
        }
    }
    myfile << "====================\nTotal rules = " << totalRules << endl;
    myfile.close();
}

void test(string mode, const vector<vector<string>> &datas, float minSupport = 0.1, float confidence = 0.2)
{
    int mSup = int(minSupport * datas.size());
    vector<assoInfo> freqItemSet;

    cout << "starting test / mode = " << mode << "\n";
    cout << "data count:" << datas.size() << "\n";
    cout << "minSup: " << mSup << "\n";

    // count time
    auto start = chrono::steady_clock::now();

    if (mode == "fp")
    {
        fpTree tree(mSup);
        tree.buildTree(datas);
        freqItemSet = tree.fpMining(assoInfo());
    }
    else if (mode == "ap")
    {
        freqItemSet = apriori(datas, mSup);
    }
    else
    {
        cout << "mode error\n";
        return;
    }

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << chrono::duration<double, milli>(diff).count() << " ms" << endl;

    if (mode == "fp")
    {
        printResult(freqItemSet, "./fp_result.txt", "./fp_rule.txt", minSupport, confidence, datas.size());
    }
    else if (mode == "ap")
    {
        printResult(freqItemSet, "./ap_result.txt", "./ap_rule.txt", minSupport, confidence, datas.size());
    }
}

int main()
{

    auto ibmData = readIBMData("./dataset/IBM5000.txt");

    // parameters: mode ( fp or ap ) , data , support , confidence
    test("fp", ibmData, 0.1, 0.2);
    test("ap", ibmData, 0.1, 0.2);

    /* auto kaggleData = readKaggleData("./dataset/kaggle.txt");
    test("fp", kaggleData, 0.006, 0.05);
    test("ap", kaggleData, 0.006, 0.05);*/
}