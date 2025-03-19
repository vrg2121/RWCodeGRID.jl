# DataAdjustments.jl

module DataAdjustments

export Matshifter

function Matshifter(M)
    K, J = size(M)

    for k = 1:K
        counter = 0
        for j = 1:J
            if counter == 1 && M[k, j] > 0
                M[k, j] = -1
            end
            
            if M[k, j] > 0
                counter += 1
            end
        end
    end

    return M
end


end