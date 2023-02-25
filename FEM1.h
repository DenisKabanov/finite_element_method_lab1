#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//Mesh related classes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//Finite element implementation classes
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
//Standard C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace dealii;

template <int dim> // dim - размерность задачи (в первой лабе всегда = 1)
class FEM
{
 public:
  //Class functions
  FEM (unsigned int order,unsigned int problem); // конструктор класса (принимает порядок базисных функций и номер задачи)
  ~FEM(); // деструктор класса

  //Function to find the value of xi at the given node (using deal.II node numbering)
  double xi_at_node(unsigned int dealNode); // позволяет узнать значение локальной координаты xi (кси) по номеру узла (из-за причудливой нумерации узлов в deal.II)
  // нумерация узлов от нуля, первый узел - самый левый узел элемента, !второй! - самый правый в конечном элементе, последующие - середина в конечном элементе (и так далее на другие конечные элементы)
  // xi_at_node(0) = -1 (в конечном элементе самая левая кси = -1, правая = 1)
  // xi_at_node(2) = -1/3; i_at_node(3) = 1/3 для случая с кубическими базисными функциями
  // нужна для следующих двух функций

  //Define your 1D basis functions and derivatives
  double basis_function(unsigned int node, double xi); // определение базисных функций Лагранжа
  double basis_gradient(unsigned int node, double xi); // определение производных базисных функции Лагранжа

  //Solution steps (вызываются в main)
  void generate_mesh(unsigned int numberOfElements); // (создание сетки, расчётная область будет разбита, например, на 10 равных по длине конечных элементов)
  void define_boundary_conds();
  void setup_system(); // изменение размеров матриц, векторов, определение параметров квадратурных формул наивысшей степени точности для нахождения определённого интеграла 
  void assemble_system(); // ассемблирование (переход от суммирования по конечным элементам к матричной записи (умножение матриц))
  void solve(); // решение системы линейных уравнений, к которым сводится задача
  void output_results(); // вывод результатов (ничего не надо менять?)

  //Function to calculate the l2 norm of the error in the finite element sol'n vs. the exact solution
  double l2norm_of_error(); // вывод нормы ошибки (мера того, на скоько полученное конечно-элементное решение отличается от точного аналитического)

  //Class objects
  Triangulation<dim>   triangulation; //mesh (конечно-элементная сетка, deal.II-ое представление)
  FESystem<dim>        fe; 	      //FE element (отдельный конечный элемент из нескольких узлов, dim=1 в первой лабе)
  DoFHandler<dim>      dof_handler;   //Connectivity matrices (объект связи локальных степеней свободы и глобальных, нужен для перехода от локальной нумерации узлов к глобальной?)

  // квадратурные формулы Гаусса
  //Gaussian quadrature - These will be defined in setup_system()
  unsigned int	        quadRule;    //quadrature rule, i.e. number of quadrature points (число точек, в которых вычисляется значение функции для нахождения значения определённого интеграла)
  // см википедию для ввода следующих значений в соответствующую функцию setup_system() по выбранному quadRule (сами выбираем, оценивая порядок фигурирующих многочленов в конечном решении, чтобы многочлены интегрировались точно, без погрешности — это наша задача, не надо переусердствовать, нужно минимальное значение quadRule)
  // если N - число точек, по которым вычисляется квадратура, то будет интегрирваться точно, без погрешности все полиномы до степени 2n-1
  std::vector<double>	quad_points; //vector of Gauss quadrature points (точки, корни многочлена Лежандра)
  std::vector<double>	quad_weight; //vector of the quadrature point weights (веса, подобраны для точного интегрирования всех полиномов до степени 2n-1)
    
  //Data structures
  SparsityPattern       sparsity_pattern; //Sparse matrix pattern (информация о том, какие элементы в разреженной матрице k отличны от нуля)
  SparseMatrix<double>  K;		 //Global stiffness (sparse) matrix
  Vector<double>        D, F; 		 //Global vectors - Solution vector (D) and Global force vector (F) (D - степени свободы, F - вектор сил из библиотеки deal.II)
  std::vector<double>   nodeLocation;	 //Vector of the x-coordinate of nodes by global dof number (вектор x координат узлов, индекс - глобальный номер узла, значение элемента вектора - x координата)
  std::map<unsigned int,double> boundary_values;	//Map of dirichlet boundary conditions (контейнер-отображение(map или словарь), содержащий граничные условия Дирихле)
  // boundary_values[0] = 0.0; boundary_values[например, 4] = например, 0.01
  double                basisFunctionOrder, prob, L, g1, g2; // basisFunctionOrder - порядок базисных функций, prob - решаемая задача (1 или 2), L - длина рассчётной области, g1 - значение Дирихле на левом конце, g2 - значение Дирихле на правом конце

  //solution name array (что-то для вывода)
  std::vector<std::string> nodal_solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
};

// Class constructor for a vector field
template <int dim>
FEM<dim>::FEM(unsigned int order,unsigned int problem)
: // объявление конечных элементов (Finite Element)
fe (FE_Q<dim>(order), dim), // FE_Q отвечает за распределение узлов, ему нужен порядок базисных функций Лагранжа, dim - размерность (в первой лабе = 1)
  dof_handler (triangulation)
{
  basisFunctionOrder = order; // указание порядка базисной функции
  if(problem == 1 || problem == 2){ // указание решаемой задачи
    prob = problem;
  }
  else{
    std::cout << "Error: problem number should be 1 or 2.\n";
    exit(0);
  }

  //Nodal Solution names - this is for writing the output file
  for (unsigned int i=0; i<dim; ++i){
    nodal_solution_names.push_back("u");
    nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  }
}

// деструктор класса
template <int dim>
FEM<dim>::~FEM(){
  dof_handler.clear(); // очищение dof_handler
}

//Find the value of xi at the given node (using deal.II node numbering)
template <int dim>
double FEM<dim>::xi_at_node(unsigned int dealNode){ // преобразование deal.II узлов в значение локальной координаты кси
  double xi;

  if(dealNode == 0){
    xi = -1.;
  }
  else if(dealNode == 1){
    xi = 1.;
  }
  else if(dealNode <= basisFunctionOrder){
    xi = -1. + 2.*(dealNode-1.)/basisFunctionOrder;
  }
  else{
    std::cout << "Error: you input node number "
	      << dealNode << " but there are only " 
	      << basisFunctionOrder + 1 << " nodes in an element.\n";
    exit(0);
  }

  return xi;
}

//Define basis functions
// Реализовать функцию, зная xi (кси) и node (номер узла)
template <int dim>
double FEM<dim>::basis_function(unsigned int node, double xi){ // A - node, xi - кси, для xi_B поможет функция xi_at_node; 
//N_A = произведение(B от 1 до числа узлов в элементе, B!=A) (xi-xi_B) / произведение(B от 1 до числа узлов в элементе, B!=A) (xi_A-xi_B)

  /*"basisFunctionOrder" defines the polynomial order of the basis function,
    "node" specifies which node the basis function corresponds to, 
    "xi" is the point (in the bi-unit, or local, domain) where the function is being evaluated.
    You need to calculate the value of the specified basis function and order at the given quadrature pt.*/

  double value = 1.; //Store the value of the basis function in this variable

  /*You can use the function "xi_at_node" (defined above) to get the value of xi (in the bi-unit domain)
    at any node in the element - using deal.II's element node numbering pattern.*/

  //EDIT_DONE
  //цикл по узлам в эл-те, n_ne - число узлов в эл-те=order+1
  for (unsigned int B=0; B<basisFunctionOrder+1; B++){
    if (B != node) {
      value *= (xi - xi_at_node(B)) / (xi_at_node(node) - xi_at_node(B));
    }
  }

  // вывод значения базисной функции
  // std::cout << "basis function at node(A): " << node << " xi: " << xi << " value: " << value << std::endl; // !!!!!!
  
  return value;
}

//Define basis function gradient
// аналогично предыдущей функции, но с производными
template <int dim>
double FEM<dim>::basis_gradient(unsigned int node, double xi){
  /*"basisFunctionOrder" defines the polynomial order of the basis function,
    "node" specifies which node the basis function corresponds to, 
    "xi" is the point (in the bi-unit domain) where the function is being evaluated.
    You need to calculate the value of the derivative of the specified basis function and order at the given quadrature pt.
    Note that this is the derivative with respect to xi (not x)*/

  double value = 0.; //Store the value of the gradient of the basis function in this variable

  /*You can use the function "xi_at_node" (defined above) to get the value of xi (in the bi-unit domain)
    at any node in the element - using deal.II's element node numbering pattern.*/

  //EDIT_DONE_?

  switch(int(basisFunctionOrder)){  //basisFunctionOrder - max B, node - A
    case 1:  // A!=B, А=1    для линейных базисных функций существует 2 node (в коде нумерация с 0 - узлы 0 и 1, а в лекциях с 1 - узлы 1 и 2)
      switch(node){
        case 0:
          value = -1./2;
          break;
        case 1:
          value = 1./2;
          break;
      }
      break;
    case 2:   // для квадратичных базисных функций
      switch(node){
        case 0:
          value = xi-1./2;
          break;
        case 1:
          value = -2*xi;
          break;
        case 2:
          value = xi+1./2;
          break;
      }
      break;
    case 3:
      switch(node){
        case 0:
          value = -27./16 * pow(xi, 2) + 9./8 * xi + 1./16;
          break;
        case 1:
          value = 81./16 * pow(xi, 2) - 9./8 * xi - 27./16;
          break;
        case 2:
          value = -81./16 * pow(xi, 2) - 9./8 * xi + 27./16;
          break;
        case 3:
          value = 27./16 * pow(xi, 2) + 9./8 * xi - 1./16;
          break;
      }
      break;
  }
//  std::cout << "returning value: " << value << std::endl; // !!!!!
  return value;
}

//Define the problem domain and generate the mesh
template <int dim>
void FEM<dim>::generate_mesh(unsigned int numberOfElements){

  //Define the limits of your domain
  L = 0.1; //EDIT_DONE_??? (в записи 1?, а в задаче 0.1)
  double x_min = 0.; // слева координата 0
  double x_max = L; // справа координата L

  // вызов deal.II функций для создания сетки
  Point<dim,double> min(x_min),
    max(x_max);
  std::vector<unsigned int> meshDimensions (dim,numberOfElements);
  GridGenerator::subdivided_hyper_rectangle (triangulation, meshDimensions, min, max);
}

//Specify the Dirichlet boundary conditions
template <int dim>
void FEM<dim>::define_boundary_conds(){ // определение граничных условий Дирихле (сложная из-за нумерации узлов в deal.II, не надо редактировать)
  const unsigned int totalNodes = dof_handler.n_dofs(); //Total number of nodes (глобально)

  //Identify dirichlet boundary nodes and specify their values.
  //This function is called from within "setup_system"

  /*The vector "nodeLocation" gives the x-coordinate in the real domain of each node,
    organized by the global node number.*/

  /*This loops through all nodes in the system and checks to see if they are
    at one of the boundaries. If at a Dirichlet boundary, it stores the node number
    and the applied displacement value in the std::map "boundary_values". Deal.II
    will use this information later to apply Dirichlet boundary conditions.
    Neumann boundary conditions are applied when constructing Flocal in "assembly"*/
  for(unsigned int globalNode=0; globalNode<totalNodes; globalNode++){
    if(nodeLocation[globalNode] == 0){ // левая граница
      boundary_values[globalNode] = g1;
    }
    if(nodeLocation[globalNode] == L){ // правая граница
      if(prob == 1){
	      boundary_values[globalNode] = g2;
      }
    }
  }
			
}

//Setup data structures (sparse matrix, vectors)
template <int dim>
void FEM<dim>::setup_system(){

  //Define constants for problem (Dirichlet boundary values)
  g1 = 0; g2 = 0.001; //EDIT_DONE_??? (значений граничных условий Дирихле из задания)

  //Let deal.II organize degrees of freedom
  dof_handler.distribute_dofs (fe); // функция, осуществляющая отслеживание глобальных и локальных степеней свободы

  //Enter the global node x-coordinates into the vector "nodeLocation"
  // заполнение массива nodeLocation
  MappingQ1<dim,dim> mapping;
  std::vector< Point<dim,double> > dof_coords(dof_handler.n_dofs());
  nodeLocation.resize(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords);
  for(unsigned int i=0; i<dof_coords.size(); i++){
    nodeLocation[i] = dof_coords[i][0];
  }

  //Specify boundary condtions (call the function)
  define_boundary_conds(); // определение граничных условий

  //Define the size of the global matrices and vectors
  sparsity_pattern.reinit (dof_handler.n_dofs(), dof_handler.n_dofs(),
			   dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  sparsity_pattern.compress();
  // подготовительная работа, изменение размеров объектов
  K.reinit (sparsity_pattern);
  F.reinit (dof_handler.n_dofs()); // вектор из deal.II (для изменения размера нельзя вызвать векторную функцию resize)
  D.reinit (dof_handler.n_dofs()); // вектор из deal.II

  //Define quadrature rule
  /*A quad rule of 2 is included here as an example. You will need to decide
    what quadrature rule is needed for the given problems*/
  // ЗАДАЧА - ПРАВИЛЬНО ОПРЕДЕЛИТЬ quadRule

  //==============================================СЛУЧАЙ С quadRule - n_int=2
  quadRule = 2; //EDIT - Number of quadrature points along one dimension (нам этого будет мало, quadRule = 2 - точное интегрирование вплоть до многочленов третьей степени)
  //в лекциях quadRule - n_int (стр.8)
  quad_points.resize(quadRule); quad_weight.resize(quadRule);

  // задание точек для вычисления значения функция при интегрировании (точки - корни многочленов Лежандра соответствующей степени, в данном примере - второй)
  quad_points[0] = -sqrt(1./3.); //EDIT
  quad_points[1] = sqrt(1./3.); //EDIT

  // веса (википедия, находятся из условия точного подсчёта интегралов до определённой степени)
  quad_weight[0] = 1.; //EDIT
  quad_weight[1] = 1.; //EDIT
  //==============================================
  // quadRule = 3; //EDIT - Number of quadrature points along one dimension (нам этого будет мало, quadRule = 2 - точное интегрирование вплоть до многочленов третьей степени)
  // //в лекциях quadRule - n_int (стр.8)
  // quad_points.resize(quadRule); quad_weight.resize(quadRule);

  // // задание точек для вычисления значения функция при интегрировании (точки - корни многочленов Лежандра соответствующей степени, в данном примере - второй)
  // quad_points[0] = -sqrt(3./5.); //EDIT
  // quad_points[1] = 0.;
  // quad_points[2] = sqrt(3./5.); //EDIT

  // // веса (википедия, находятся из условия точного подсчёта интегралов до определённой степени)
  // quad_weight[0] = 5./9.; //EDIT
  // quad_weight[1] = 8./9.; //EDIT
  // quad_weight[1] = 5./9.; //EDIT
  //==============================================

  //Just some notes...
  std::cout << "   Number of active elems:       " << triangulation.n_active_cells() << std::endl;
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;   
}

//Form elmental vectors and matrices and assemble to the global vector (F) and matrix (K)
template <int dim>
void FEM<dim>::assemble_system(){ // ассемблирование (переход от суммирования по конечным элементам к матричной записи (умножение матриц))

  K=0; F=0; // K - матрица жёсткости, F - вектор сил (правая часть в матричном уравнении) // глобальные (полноразмерные)

  const unsigned int   			dofs_per_elem = fe.dofs_per_cell; //This gives you number of degrees of freedom per element (количество степеней свободы в элементе)
  // cell - элемент в deal.II
  // FullMatrix - полная матрица
  // Klocal - обычная двумерная матрица (двумерные оси)?, локальная матрица K
  // Flocal - локальная матрица F
  FullMatrix<double> 				Klocal (dofs_per_elem, dofs_per_elem);
  Vector<double>      			Flocal (dofs_per_elem);
  std::vector<unsigned int> local_dof_indices (dofs_per_elem); // количество элементов в local_dof_indices равно числу узлов в элементе (local_dof_indices связываем локальные степени свободы и глобальные, связывает одну нумерацию узлов с другой)
  // На примере трёх-узлового элемента: 
  // Глобальная нумерация узлов в элементах: (0 2 1) (1 4 3) (3 6 5)
  // Локальная нумерация узлов в элементах: (0 2 1) (0 2 1) (0 2 1)
  //    конечный элемент:   1   |  2    |  3
  // id для след вектора: 0 1 2 | 0 1 2 | 0 1 2
  //   local_dof_indices: 0 1 2 | 1 3 4 | 3 5 6
  double										h_e, x, f;
  // h_e - длина элемента; x - соответствует кси (нужен для вычисления f); f - значение f

  //цикл по элементам
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), 
  // elem - итератор, указывающий на начало структуры dof_handler; endc - указывает на элемент, что следует сразу за последним элементом
    endc = dof_handler.end();
  for (;elem!=endc; ++elem){ // цикл, пока текущий элемент не будет указывать на следующий за последним

    /*Retrieve the effective "connectivity matrix" for this element
      "local_dof_indices" relates local dofs to global dofs,
      i.e. local_dof_indices[i] gives the global dof number for local dof i.*/
    elem->get_dof_indices (local_dof_indices); // получение матрицы связности для конкретного элемента

    /*We find the element length by subtracting the x-coordinates of the two end nodes
      of the element. Remember that the vector "nodeLocation" holds the x-coordinates, indexed
      by the global node number. "local_dof_indices" gives us the global node number indexed by
      the element node number.*/
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]]; // длина элемента (local_dof_indices[0] - начало элемента, local_dof_indices[1] - конец элемента)

    //Loop over local DOFs and quadrature points to populate Flocal and Klocal.
    // интегрирование правой части (находим Flocal и Klocal и переходим к глобальным матрицам)
    Flocal = 0.;  //считаем, что все элементы вектора равны 0
    // Flocal[i] = A*h_e/2 * int(от -1 до 1)(N_i(xi)f(x(xi)))dxi =
    // = A*h_e/2 * summ(j от 1 до quadRule)(N_i(xi_j) * f(x(xi_j)) * weight[j])
    // N_i(xi_j) - базисная функция (считается с помощью функции basis_function?)
    // f(x(xi_j)) = f(xi_j)?
    // считаем, что А=1
    for(unsigned int A=0; A<dofs_per_elem; A++){// цикл по всем узлам эл-та (dofs_per_elem - количество степеней свободы в элементе)
      for(unsigned int q=0; q<quadRule; q++){ // суммирование по квадратуре
        x = 0;
        //Interpolate the x-coordinates at the nodes to find the x-coordinate at the quad pt.
        for(unsigned int B=0; B<dofs_per_elem; B++){ // интерполяция для отображения локальных координат xi в глобальный x
          // (преобразуем кси в икс, пользуясь изопараметрической формулировкой)

//          std::cout << "real x location: " << nodeLocation[local_dof_indices[B]] << std::endl; // !!!!!
          x += nodeLocation[local_dof_indices[B]]*basis_function(B, quad_points[q]); // для подсчёта f(x(xi_j))
          //nodeLocation - берет X_e_A, 
          //local_dof_indices мапит локальный номер узла к глобальному, а nodeLocation хранит координаты х 
        }
//        std::cout << "A: " << A << "; x with respect to Legandre root: " << x << std::endl; // !!!!!
        //EDIT_DONE_? - Define Flocal.
        // надо определить Flocal, используя квадратуру Гаусса для нахождения интеграла
        //согласно заданию, F(x) = f = 10^11Нм^(−4)*x, Нм - Ньютон на метр
        //long long pow(10, 11)
        Flocal[A] += basis_function(A, quad_points[q]) * pow(10, 11) * x * quad_weight[q];
      }
      Flocal[A] *= h_e/2;
    }

//    вывод Flocal для элемента elem !!!!!
//    for(int A=0; A<dofs_per_elem;A++){// !!!!!
//      std::cout << "elem: " << A << " Flocal[elem]: " << Flocal[A] << std::endl;// !!!!!
//    } // !!!!!

    //Add nonzero Neumann condition, if applicable
    // если задача имеет номер 2, то используем это условие для определения правой части (вкладываем его в вектор F)
    if(prob == 2){ 
      if(nodeLocation[local_dof_indices[1]] == L){
	    //EDIT_DONE_? - Modify Flocal to include the traction on the right boundary.
        Flocal[dofs_per_elem - 1]  += pow(10, 10);  // -1, т.к. индексы с 0, а нумерация узлов с 1, (индекс 0 - 1й узел) ??? (граничное условие Неймана, добавляем tA к последнему элементу вектора сил, в задаче A=1, t=h=10^10 Ньютон*метр^-2)
      }
    }

    //Loop over local DOFs and quadrature points to populate Klocal
    Klocal = 0;
    for(unsigned int A=0; A<dofs_per_elem; A++){
      for(unsigned int B=0; B<dofs_per_elem; B++){
        for(unsigned int q=0; q<quadRule; q++){
          //EDIT_DONE - Define Klocal.
          // вставить код для определения компонентов Klocal (применить квадратурные формулы Гаусса)
          // Klocal[i][j] = int(от -1 до 1) (N_i'xi * N_j'xi) dxi
          // std::cout << basis_gradient(A, quad_points[q]) * basis_gradient(B, quad_points[q]) * quad_weight[q] << std::endl;// !!!!!
          Klocal.add(A, B, 2. * pow(10,11) / h_e * basis_gradient(A, quad_points[q]) * basis_gradient(B, quad_points[q]) * quad_weight[q]);
        }
      }
    }
    // for(int a=0; a<dofs_per_elem; a++){// !!!!!
    //   for(int b=0; b<dofs_per_elem;b++){// !!!!!
    //     std::cout << Klocal[a][b] << "\t";// !!!!!
    //   } // !!!!!
    //   std::cout << std::endl;// !!!!!
    // } // !!!!!

    //Assemble local K and F into global K and F
    //You will need to used local_dof_indices[A]
    // Ассамблирование - переход от локальных матриц к глобальным
    // Важно помнить, что K - sparse (разреженная) матрица, поэтому нельзя просто написать K[i][j], используется команда K.add
    //*приводить матрицу K к квадратному виду (в задаче Дирихле) здесь не нужно, это делает deal.II с помощью apply_boundary_values
    for(unsigned int A=0; A<dofs_per_elem; A++){
      //EDIT_DONE_? - add component A of Flocal to the correct location in F
      /*Remember, local_dof_indices[A] is the global degree-of-freedom number corresponding to element node number A*/
      F[local_dof_indices[A]] += Flocal[A];

      for(unsigned int B=0; B<dofs_per_elem; B++){
        //EDIT_DONE_? - add component A,B of Klocal to the correct location in K (using local_dof_indices)
        /*Note: K is a sparse matrix, so you need to use the function "add".
          For example, to add the variable C to K[i][j], you would use:
          K.add(i,j,C);*/
          // std::cout<<local_dof_indices[A] << " " << local_dof_indices[B] << " " << Klocal[A][B]<< std::endl; // !!!!!
          K.add(local_dof_indices[A], local_dof_indices[B], Klocal[A][B]);
      }
    }
  }

  // Вывод матрицы K !!!!!
  std::cout << "Матрица K:" << std::endl;
  K.print(std::cout, false, false);

  // Вывод вектора F !!!!!
  std::cout << "Вектор F:" << std::endl;
  for (int i = 0; i < K.get_sparsity_pattern().n_rows(); i++)
    std::cout << F[i] << "\t";
  std::cout << std::endl;

  //Apply Dirichlet boundary conditions
  /*deal.II applies Dirichlet boundary conditions (using the boundary_values map we
    defined in the function "define_boundary_conds") without resizing K or F*/
  MatrixTools::apply_boundary_values (boundary_values, K, D, F, false);

  // Вывод матрицы K !!!!!
  // K.print(std::cout, false, false);

  // Вывод вектора F !!!!!
  // std::cout << "Вектор F:" << std::endl;
  // for (int i = 0; i < K.get_sparsity_pattern().n_rows(); i++)
  //   std::cout << F[i] << "\t";
  // std::cout << std::endl;
}

//Solve for D in KD=F
template <int dim>
void FEM<dim>::solve(){

  //Solve for D
  SparseDirectUMFPACK  A; // разреженная матрица для конкретного решателя UMFPACK
  A.initialize(K); // инициализируем матрицу A с помощью матрицы K
  A.vmult (D, F); //D=K^{-1}*F (получаем вектор-столбец D)

}

//Output results
template <int dim>
void FEM<dim>::output_results (){ // вывод в VTK файлы (ничего не надо менять)

  //Write results to VTK file
  std::ofstream output1("solution.vtk");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  //Add nodal DOF data
  data_out.add_data_vector(D, nodal_solution_names, DataOut<dim>::type_dof_data,
			   nodal_data_component_interpretation);
  data_out.build_patches();
  data_out.write_vtk(output1);
  output1.close();
}

template <int dim>
double FEM<dim>::l2norm_of_error(){ // функция подсчёта l2 ошибки (итерирование схоже функции ассемблирования)
	
  double l2norm = 0.;

  //Find the l2 norm of the error between the finite element sol'n and the exact sol'n
  const unsigned int   			dofs_per_elem = fe.dofs_per_cell; //This gives you dofs per element
  std::vector<unsigned int> local_dof_indices (dofs_per_elem);
  double u_exact, u_h, x, h_e;

  //loop over elements  
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active (), 
    endc = dof_handler.end();
  for (;elem!=endc; ++elem){

    //Retrieve the effective "connectivity matrix" for this element
    elem->get_dof_indices (local_dof_indices);

    //Find the element length
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];

    for(unsigned int q=0; q<quadRule; q++){ // находим интеграл
    	// l2 норма ошибки (без указания корня) = int(от 0 до 1) (u_h - u)^2 dx = [пока не контролируем производную, u - точное решение, u_h - конечно-элементное решение] =
      // = summ(по конечным элементам) (int(область конечного элемента _/\_e) (u_h - u)^2 dx) = [замена переменной x на xi в интеграле]
      // = summ(по конечным элементам) (int(от -1 до 1) ((u_h - u)^2 * h_e/2) dxi) 
      //==> проблема - вычислять u_h в произвольном xi (мы можем вычислять u_h в xi, что соответствует узлам, 
      //но чтобы вычислять u_h в произвольной xi нужно, как при вычислении правой части - F, посчитать x по xi (изопараметрическое задание))
      x = 0.; u_h = 0.;
      //Find the values of x and u_h (the finite element solution) at the quadrature points
      for(unsigned int B=0; B<dofs_per_elem; B++){
        // преобразуем кси в икс, пользуясь изопараметрической формулировкой
        x += nodeLocation[local_dof_indices[B]] * basis_function(B, quad_points[q]); // для подсчёта u_h в произвольном xi
        u_h += D[local_dof_indices[B]] * basis_function(B, quad_points[q]);
        // восстанавливаем u_h только в тех точках, что нам нужны (зная степени сводобы local_dof_indices[B], так как уже решили систему (нашли D), и используя базисные функции)
      }
      //EDIT_DONE_? - Find the l2-norm of the error through numerical integration.
      /*This includes evaluating the exact solution at the quadrature points*/

      double dudx0 = 0.; // значение сигмы, делённой на E в нуле 

      if (prob == 1) { // задача Дирихле-Дирихле
        // dudx0 = (g2 + pow(10,11) * pow(L, 3) / (6 * pow(10,11)) - g1) / L;
        dudx0 = (g2 + pow(L, 3) / 6  - g1) / L; // убрали повторяющиеся значения pow(10,11)
      } else {
        // dudx0 = ;
      }

      // подсчёт аналитического решения (см рисовалки в paint, аналитическое решение 2)
      // u_exact = dudx0 * x - pow(10,11) * pow(x, 3) / (6 * pow(10,11)) + g1;
      u_exact = dudx0 * x - pow(x, 3) / 6  + g1; // убрали повторяющиеся значения pow(10,11)

      // l2norm += (pow(u_h,2) - 2 * u_exact * u_h + pow(u_exact,2)) * quad_weight[q] * h_e / 2; //по квадратурной формуле Гаусса
      l2norm += pow((u_h - u_exact), 2) * quad_weight[q] * h_e / 2; //по квадратурной формуле Гаусса
    }
  }

  return sqrt(l2norm);
}
