#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "FEM1.h"
#include "writeSolutions.h"

using namespace dealii;

int main (){
  try{
    deallog.depth_console (0); // служебная команда (должна быть)

		//Specify the basis function order: 1, 2, or 3
		unsigned int order = 1; // порядок базисных функций Лагранжа == число узлов в элементе - 1(используется в представлении пробной и весовой функциях)

		//Specify the subproblem: 1 or 2
		unsigned int problem = 1; // номер задачи (1 - задача Дирихле-Дирихле, в точке 0 и L заданы условия Дирихле; 2 - задача Дирихле-фон Неймана)

    FEM<1> problemObject(order,problem); // создание объекта проблемы (как раз надо реализовать его методы); 1 - размерность задачи (в первой лабе всегда 1)
    
    std::cout << "Object created" << std::endl;

    //Define the number of elements as an input to "generate_mesh"
    problemObject.generate_mesh(10); //e.g. a 10 element mesh (создание сетки, расчётная область будет разбита на 10 равных по длине конечных элементов)
    std::cout << "generate_mesh" << std::endl;
    problemObject.setup_system(); // изменение размеров матриц, векторов, определение параметров квадратурных формул наивысшей степени точности для нахождения определённого интеграла 
    std::cout << "setup_system" << std::endl;
    problemObject.assemble_system(); // ассемблирование (переход от суммирования по конечным элементам к матричной записи (умножение матриц))
    std::cout << "assemble_system" << std::endl;

// K, F, prob, basisFunctionOrder, L, g1, g2
    std::cout << "prob: " << problemObject.prob << std::endl;
    std::cout << "basisFunctionOrder: " << problemObject.basisFunctionOrder << std::endl;
    std::cout << "L: " << problemObject.L << std::endl;
    std::cout << "g1: " << problemObject.g1 << std::endl;
    std::cout << "g2: " << problemObject.g2 << std::endl;
    std::cout << "SparseMatrix n_rows: " << problemObject.K.get_sparsity_pattern().n_rows() << std::endl;
    std::cout << "SparseMatrix n_cols: " << problemObject.K.get_sparsity_pattern().n_cols() << std::endl;
    // std::cout << "K: " << problemObject.K << std::endl; // error: no match for ‘operator<<’
    // std::cout << MatrixXd(problemObject.K) << std::endl;
    // std::cout << "get_sparsity_pattern (0,0): " << problemObject.K.get_sparsity_pattern()(0,0) << std::endl;
    // std::cout << "get_sparsity_pattern (0,1): " << problemObject.K.get_sparsity_pattern()(0,1) << std::endl;
    // std::cout << "get_sparsity_pattern (1,0): " << problemObject.K.get_sparsity_pattern()(1,0) << std::endl;
    // std::cout << "get_sparsity_pattern (1,1): " << problemObject.K.get_sparsity_pattern()(1,1) << std::endl;

    problemObject.solve(); // решение системы линейных уравнений, к которым сводится задача
    std::cout << "solve" << std::endl;
    std::cout << problemObject.l2norm_of_error() << std::endl; // вывод нормы ошибки (мера того, на скоько полученное конечно-элементное решение отличается от точного аналитического)
    
    //write output file in vtk format for visualization
    problemObject.output_results(); // вывод результатов (ничего не надо менять?)
    std::cout << "output_results" << std::endl;
    
    //write solutions to h5 file (vtk файлы, что можно просмотреть с помощью программы paraview)
    char tag[21];
    sprintf(tag, "CA1_Order%d_Problem%d",order,problem);
    writeSolutionsToFileCA1(problemObject.D, problemObject.l2norm_of_error(), tag);
  }
  catch (std::exception &exc){
    std::cerr << std::endl << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
	      << exc.what() << std::endl
	      << "Aborting!" << std::endl
	      << "----------------------------------------------------"
	      << std::endl;

    return 1;
  }
  catch (...){
    std::cerr << std::endl << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
	      << "Aborting!" << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    return 1;
  }

  return 0;
}
