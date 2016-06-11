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
            if nargin >0;
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
            
            syms a b;
            smin = symfun(ge(a,b)^1*a + (1-ge(a,b)^1)*b,[a b]);
            nlog = symfun(log(b)/log(a), [a,b]);
            Aand = symfun(a*b,[a b]);
            Aor  = symfun(a+b-a*b,[a b]);
            Yand = symfun(1-smin(1, ((1-a)^p+(1-b)^p)^(1/p)),[a b]);
            Yor  = symfun(smin(1,(a^p+b^p)^(1/p)),[a b]);
            Dand = symfun(1/((1+(1/a-1)^p+(1/b-1)^p)^(1/p)),[a b]);
            Dor  = symfun(1/((1+(1/a-1)^-p+(1/b-1)^-p)^-(1/p)),[a b]);
            Hand = symfun(a*b/(p+(1-p)*(a+b-a*b)),[a b]);
            Hor  = symfun((a+b-(2-p)*a*b)/(1-(1-p)*a*b),[a b]);
            Fand = symfun(nlog(p,(1+(p^a-1)*(p^b-1)/(p-1))),[a b]);
            For  = symfun((1-nlog(p,1+((p^(1-a)-1)*(p^(1-b)-1))/(p-1))),[a b]);
            operators = containers.Map({'Algebraic','Yager','Dombi','Hamacher','Frank'},{{Aand,Aor},{Yand,Yor},{Dand,Dor},{Hand,Hor},{Fand,For}});
            op = operators(norm);
            obj.NOT = symfun(1-a, a);
            obj.AND = op{1};
            obj.OR  = op{2};
            syms sJ sQ;
            JKeq    = symfun(obj.AND(obj.OR(sJ,sQ),obj.AND(obj.OR(sJ,sQ),obj.OR(sQ,obj.NOT(sQ)))),[sJ sQ]);
            Deq     = symfun(obj.AND(obj.OR(sJ,sJ),obj.AND(obj.OR(sJ,sQ),obj.OR(sJ,obj.NOT(sQ)))), [sJ sQ]);
            ChoiDeq = symfun(obj.AND(sJ,obj.AND(obj.OR(sJ,sQ),obj.OR(obj.NOT(sQ),sJ))), [sJ sQ]);
            types = containers.Map({'JK','D','ChoiD'},{JKeq,Deq,ChoiDeq});
            obj.fundamentalEQ = matlabFunction(types(type));
            end;
        end;
        
        function response = activation_function(obj,X)
            
            response = obj.fundamentalEQ(logsig(X),obj.Q);
            if obj.constQ == false
                obj.Q = response;
            end;
        end
        
        function cp = copy(obj)
            cp = feval(class(obj));
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

