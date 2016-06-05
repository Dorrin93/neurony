classdef FFNeuron < BaseNeuron
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        NOT;
        AND;
        OR;
        Q;
        fundamentalEQ;
        constQ;
    end
    
    properties(Access = private)
        type;
        norm;
    end
    
    methods
        %%
        % -<REQ>type - Typ flip-flopa [ 'JK','D','ChoiD']
        % -<REQ>norm - Norma ['Algebraic','Yager','Dombi','Hamacher','Frank']
        % -p - Parametr do norm przy braku jest ustalany na p=2
        % -q - Początkowa wartość Q, przy braku jest losowana (0,1]
        % -constQ - flaga odpowiadająca za to czy Q ma być ustalone, czy
        % zmieniane w kolejnych wywołaniach, przy braku jest constQ=false
        %%
        function obj = FFNeuron(type,norm,p,q,constQ)
            obj@BaseNeuron();
            if nargin < 3;
                p = 2;
            end;
            
            obj.type = type;
            obj.norm = norm;
            
            if nargin < 4;
                obj.Q = rand();
            else
                obj.Q = q;
            end;
            
            if nargin < 5;
                obj.constQ = false;
            else
                obj.constQ = constQ;
            end;
            
            nlog = @(a,b) log(b)/log(a);
            Aand = @(a,b) a*b;
            Aor  = @(a,b) a+b-a*b;
            Yand = @(a,b) 1-min([1; ((1-a)^p+(1-b)^p)^(1/p)]);
            Yor  = @(a,b) min([1;(a^p+b^p)^(1/p)]);
            Dand = @(a,b) 1/((1+(1/a-1)^p+(1/b-1)^p)^(1/p));
            Dor  = @(a,b) 1/((1+(1/a-1)^-p+(1/b-1)^-p)^-(1/p));
            Hand = @(a,b) a*b/(p+(1-p)*(a+b-a*b));
            Hor  = @(a,b) (a+b-(2-p)*a*b)/(1-(1-p)*a*b);
            Fand = @(a,b) nlog(p,(1+(p^a-1)*(p^b-1)/(p-1)));
            For  = @(a,b) (1-nlog(p,1+((p^(1-a)-1)*(p^(1-b)-1))/(p-1)));
            operators = containers.Map({'Algebraic','Yager','Dombi','Hamacher','Frank'},{{Aand,Aor},{Yand,Yor},{Dand,Dor},{Hand,Hor},{Fand,For}});
            op = operators(norm);
            obj.NOT = @(a) 1-a;
            obj.AND = op{1};
            obj.OR  = op{2};
            JKeq    = @(J,Q) obj.AND(obj.OR(J,Q),obj.AND(obj.OR(J,Q),obj.OR(Q,obj.NOT(Q))));
            Deq     = @(J,Q) obj.AND(obj.OR(J,J),obj.AND(obj.OR(J,Q),obj.OR(J,obj.NOT(Q))));
            ChoiDeq = @(J,Q) obj.AND(J,obj.AND(obj.OR(J,Q),obj.OR(obj.NOT(Q),J)));
            types = containers.Map({'JK','D','ChoiD'},{JKeq,Deq,ChoiDeq});
            obj.fundamentalEQ = types(type);
        end;
        
        function response = activation_function(obj,X)
            
            response = obj.fundamentalEQ(X,obj.Q);
            if obj.constQ == false
                obj.Q = response;
            end;
        end
        
        function cp = copy(obj)
            cp = feval(class(obj), obj.type, obj.norm);
            cp.type = obj.type;
            cp.norm = obj.norm;
            cp.weights = obj.weights;
            cp.activation_function_parameters = obj.activation_function_parameters;
            cp.bias = obj.bias;
            cp.NOT = obj.NOT;
            cp.AND = obj.AND;
            cp.OR = obj.OR;
            cp.Q = obj.Q;
            cp.fundamentalEQ = obj.fundamentalEQ;
            cp.constQ = obj.constQ;
        end
        
    end
    
end

