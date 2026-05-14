// Parallel BFS and DFS using OpenMP with User Input

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph {
public:
    int nodes;
    vector<vector<int>> adjList;

    Graph(int n) {
        nodes = n;
        adjList.resize(n);
    }

    // Add edge in undirected graph
    void addEdge(int u, int v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    // Parallel BFS
    void parallelBFS(int start) {
        vector<bool> visited(nodes, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "\nParallel BFS Traversal: ";

        while (!q.empty()) {
            int size = q.size();

            #pragma omp parallel for
            for (int i = 0; i < size; i++) {

                int current = -1;

                // Critical section for queue access
                #pragma omp critical
                {
                    if (!q.empty()) {
                        current = q.front();
                        q.pop();
                        cout << current << " ";
                    }
                }

                if (current != -1) {
                    for (int neighbor : adjList[current]) {

                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }

        cout << endl;
    }

    // Parallel DFS
    void parallelDFS(int start) {
        vector<bool> visited(nodes, false);
        stack<int> s;

        s.push(start);

        cout << "\nParallel DFS Traversal: ";

        while (!s.empty()) {

            int current;

            #pragma omp critical
            {
                current = s.top();
                s.pop();
            }

            if (!visited[current]) {

                visited[current] = true;
                cout << current << " ";

                #pragma omp parallel for
                for (int i = 0; i < adjList[current].size(); i++) {

                    int neighbor = adjList[current][i];

                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                        }
                    }
                }
            }
        }

        cout << endl;
    }
};

int main() {

    int n, e;

    cout << "Enter number of nodes: ";
    cin >> n;

    Graph g(n);

    cout << "Enter number of edges: ";
    cin >> e;

    cout << "\nEnter edges (u v):\n";

    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;

    cout << "\nEnter starting node: ";
    cin >> start;

    g.parallelBFS(start);
    g.parallelDFS(start);

    return 0;
}


/*Enter number of nodes: 5
Enter number of edges: 4

Enter edges (u v):
0 1
0 2
1 3
2 4

Enter starting node: 0*/
